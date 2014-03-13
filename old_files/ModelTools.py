import casadi as cs
import numpy  as np

def parse_string(string,delimiter=' + '):
    """
    Parses a reaction string into a list of strings, each
    one a state (or group of states, for linear cooperation)
    """
    species = []
    del_len = len(delimiter)
    while True:
        try:
            ind = string.index(delimiter)
            species_str = string[:ind]
            species += [species_str]
            string = string[(ind+del_len):]
        except ValueError:
            species += [string]
            break
    return species

def unparse_string(stringlist, delimiter=' + '):
    """
    Undo the effects of parse_string, returning a list-wrapped combined
    string from a list of strings.
    """
    try:
        string_out = stringlist[0]
        for entry in stringlist[1:]:
            string_out += delimiter + entry
        return [string_out]
    except IndexError:
        return []

class States(list):
    """
    Temporary place to put state-management code. Hopefully a class such
    as this will help in reducing the model size, but for now it will be
    a place to put string lookup and creation functions.
    """

    def __init__(self, *args, **kwargs):
        super(States, self).__init__(*args,**kwargs)

    def __call__(self, species_str):
        """
        Converts a list of string state representations into
        their corresponding casadi symbolic variables.
        """

        species_csx = []
        for state in species_str:
            if state not in self.listrep(): 
                # Here we might add a state variable
                raise ValueError('State \'' + state + '\' not found')

            for i,entry in enumerate(self.listrep()):
                if state == entry: species_csx += [self[i]]

        return species_csx

    def lookup(self, state_str):
        """
        Lookup individual state and return index
        """

        if state_str not in self.listrep(): 
            raise ValueError('State \''
                             + state_str + '\' not found')

        for i, entry in enumerate(self.listrep()):
            if state_str == entry: return self[i], i

    def lookup_loose(self, state_str):
        """
        similar to self.lookup, but returns a list of all states
        where state_str is found. Returns a list of tuples, where
        tuple[0] is the cs.SX state, and tuple[1] is the parameter
        index.
        """
        return_list = []
        for i, entry in enumerate(self.listrep()):
            if state_str in entry: return_list += [(self[i], i)]
        return return_list

    def listrep(self):
        """
        Get list of strings of state ids
        """
        return [repr(entry) for entry in self]


class Parameters(list):
    """
    A class to manage the creation and naming of various kinetic
    reaction parameters
    """

    def __init__(self, *args, **kwargs):
        super(Parameters, self).__init__(*args,**kwargs)
        
        # Create dictionaries for reaction types and number of
        # parameters already of that type. This will combine to form
        # labels for parameter values.
        self.basedict = {
            'rate'  : 'k',
            'MM'    : 'Km',
            'coop'  : 'M',
            'tmax'  : 'Vm',
            'tinh'  : 'Ki',
            'tact'  : 'Ka',
            'thill' : 'n' ,
        }

        self.inddict = {}

        for key in self.basedict.iterkeys(): self.inddict[key] = 0

    def __call__(self, rtype, name=None):
        base = self.basedict[rtype]

        # if a parameter name was passed (typically a state variable
        # string, get the lowest unique (base + name + integer) name not
        # already used.
        # if name:
        #     pname = base + '_' + name
        #     if pname in self.listrep():
        #         name_ind = 1
        #         while pname in self.listrep():
        #             pname = base + '_' + name + '_' + str(name_ind)
        #             name_ind += 1
        # else:
        self.inddict[rtype] += 1
        pname = base + '_' + str(self.inddict[rtype])

        par = cs.SX(pname)
        self += [par]
        return par

    def lookup(self, parameter_str):
        """
        Equivalent to States.__call__(), returns a cs.SX object matching
        a given parameter string.
        """

        if parameter_str not in self.listrep(): 
            raise ValueError('Parameter \''
                             + parameter_str + '\' not found')

        for i, entry in enumerate(self.listrep()):
            if parameter_str == entry: return self[i], i

    def lookup_loose(self, parameter_str):
        """
        similar to self.lookup, but returns a list of all parameters
        where parameter_str is found. Returns a list of tuples, where
        tuple[0] is the cs.SX parameter, and tuple[1] is the parameter
        index.
        """
        return_list = []
        for i, entry in enumerate(self.listrep()):
            if parameter_str in entry: return_list += [(self[i], i)]
        return return_list


    def listrep(self):
        """
        Get list of strings of parameter ids
        """
        return [repr(entry) for entry in self]

    def reset(self):
        """
        clears all entries from parameter class
        """
        for key in self.basedict.iterkeys(): self.inddict[key] = 0
        while self: self.pop()


class GenericRxn(object):
    """
    Class to hold information on a single reaction in a genetic
    regulatory network. This class will be used for Mass-Action, and
    Michealis Menten (Pseudo-Steady State) reactions, and the superclass
    TranscriptionRxn will overwrite kinetic data for hill-type gene
    inputs. 

    self.states, self.params are links to model-wide classes for
    managing the collection and naming of parameters and state
    variables.
    
    self.reactants, products, activators, and inhibitors are
    lists of strings (changed from casadi objects) for keeping track
    where states appear in the reaction.

    self.rate (computed via find_rate) should be a casadi-computable
    symbolic representation of the dynamics of the reaction, complete
    with appropriate parameter values.
    """

    def __init__(self, states, params, rxnstring=None):
        """
        states, params should be model-wide classes used to manage
        variables. rxnstring input is for shorthand access of
        self.read_rxn_str(rxnstring), if left blank, the reaction can
        later be added through a seperate function call.
        """

        self.states = states # State management
        self.params = params # Parameter management
        self.reactants  = []
        self.products   = []
        self.activators = []
        self.inhibitors = []
        self.mark = '--'

        if rxnstring: self.read_rxn_str(rxnstring)


    def read_rxn_str(self, rxnstring):
        """
        Reverse of __str__, should enable shorthand input of reaction
        reactants, products, activators, and inhibitors.
        """
                
        if rxnstring[0] is not self.mark[0]: # Reactants
            reactind = rxnstring.index(' --')
            reactstr = rxnstring[:reactind]
            self.reactants = parse_string(reactstr)
            rxnstring = rxnstring[reactind+1:]

        if rxnstring[2] is '(': # Activators
            actind = rxnstring.index(')')
            actstr = rxnstring[3:actind]
            self.activators = parse_string(actstr, delimiter=', ')
            rxnstring = rxnstring[actind+1:]

        #If no activators, cut to '/'
        if '/' in rxnstring:
            rxnstring = rxnstring[rxnstring.index('/'):]

        if rxnstring[1] is '(': # Inhibitors
            inhind = rxnstring.index(')')
            inhstr = rxnstring[2:inhind]
            self.inhibitors = parse_string(inhstr, delimiter=', ')
            rxnstring = rxnstring[inhind+1:]

        # Products
        rxnstring = rxnstring[rxnstring.index('>')+1:]
        if rxnstring is not '':
            rxnstring = rxnstring[1:]
            self.products = parse_string(rxnstring)

        # Get name for parameter values
        self.paramname = ''
        if self.reactants:
            for entry in self.reactants:
                self.paramname += entry
        else:
            for entry in self.products:
                self.paramname += entry
        # # remove trailing underscore
        # self.paramname = self.paramname[:-1]


    def __str__(self):
        """
        return useful notation for reaction class
        """
        
        # Process Reactants:
        description = ''
        for reactant in self.reactants:
            if description is not '': description += ' + '
            description += reactant

        # Print Arrow
        if description is not '': description += ' '
        description += self.mark

        # Activators
        actstr = ''
        for activator in self.activators:
            if actstr is not '': actstr += ', '
            actstr += activator
        if self.activators: description += '(' + actstr + ')'

        description += '/'

        # Inhibitor
        inhstr = ''
        for inhibitor in self.inhibitors:
            if inhstr is not '': inhstr += ', '
            inhstr += str(inhibitor)
        if self.inhibitors: description += '(' + inhstr + ')'

        description += self.mark + '> '

        # Products
        prodstr = ''
        for product in self.products:
            if prodstr is not '': prodstr += ' + '
            prodstr += str(product)
        description += prodstr

        return description
    
    def process_states(self, group):
        """
        Convert from list-of-strings to computable casadi SX
        formulations, using lookup provided by state class. Will handle
        linear cooperations by adding states together (with subsequent
        states gaining a 'M' parameter

        sept 20: I'm not sure if the M parameter is the best way to do
        this. perhaps just allow linear cooperativity before hill
        kinetics. (Each state would get seperate Ki). Currently uses M
        method in plos paper.
        
        """

        # Grow with iterations
        group_expr = []

        for state in group:
            if ' + ' in state: # we have cooperativity
                state_expr = 0
                states = parse_string(state, delimiter=' + ') 
                
                # Add first state
                state_expr += self.states([states[0]])[0]

                # For subsequent states, multiply by parameter
                for substate in states[1:]:
                    state_expr += (self.states([substate])[0] *
                                   self.params('coop',
                                               name=str(substate)))
            else: # single state
                state_expr = self.states([state])[0]

            # Add this states expression to list
            group_expr += [state_expr]

        return group_expr

                

    def find_rate(self):
        """
        Finds the rate of a mass-action or michealis-menten biochemical
        reaction, given appropriate inputs in class structures. Not
        implemented are any reactions with stochiometric coefficients
        greater than 1
        """

        # convert to casadi objects
        reactants  = self.process_states(self.reactants)
        activators = self.process_states(self.activators)
        inhibitors = self.process_states(self.inhibitors)

        self.rate = (self.params('rate', name=self.paramname) *
                     np.prod(reactants)
                     * np.prod(activators))
        inh_total = 0
        for inhibitor in inhibitors:
            inh_total += inhibitor/self.params('MM', name=self.paramname)
            
        self.rate *= 1/(1 + inh_total)
        return self.rate



class TranscriptionRxn(GenericRxn):
    def __init__(self, states, params, rxnstring=None):
        super(TranscriptionRxn, self).__init__(states, params)
        self.mark = '~~'

        if rxnstring: self.read_rxn_str(rxnstring)

    def find_rate(self):
        """
        Generates input function according to general formula for
        transcription factor regulation, Alon 2007 (Book), p. 255
        """
        
        # convert to casadi objects
        activators = self.process_states(self.activators)
        inhibitors = self.process_states(self.inhibitors)

        numerator = 0
        denominator = 1

        for i, activator in enumerate(activators):
            stract = parse_string(self.activators[i], delimiter=' + ')[0]
            actrate = (self.params('tmax', name=stract) *
                       (activator /
                        self.params('tact', name=stract))**
                        self.params('thill', name=stract))
            
            numerator += actrate
            denominator += actrate

        for i, inhibitor in enumerate(inhibitors):
            strinh = parse_string(self.inhibitors[i], delimiter=' + ')[0]
            tmax = self.params('tmax', name=strinh)
            inhrate = (tmax * (inhibitor / self.params('tinh',
                       name=strinh)) ** self.params('thill',
                       name=strinh))

            numerator += tmax
            denominator += inhrate


        self.rate = numerator/denominator
        return self.rate



class Model(object):
    """
    This may at some point map +/- operators to allow easy
    addition/removal of reactions, states, and parameters.

    Could use type(input) to determine what type of element, then search
    (using listreps?) to find the element to remove. Parameters could be
    replaced by 1's, reactions could be removed, (states considered
    consitutative?).
    """

    def __init__(self, states=None):
        if not states: self.states = States()
        else: self.states = states

        self.params = Parameters()
        self.odes = []
        self.reactions = []

        self.rxnclass = {
            '--' : GenericRxn,
            '~~' : TranscriptionRxn }

    def add_reaction(self, rxnstring):
        """
        Takes a rxnstring, decides appropriate rate class, and adds
        the rate to the model using state and parameter classes
        """

        for key in self.rxnclass.iterkeys():
            if key in rxnstring: 
                rxn = self.rxnclass[key](self.states, self.params)
                rxn.read_rxn_str(rxnstring)
                self.reactions += [rxn]
                return
        raise KeyError('Unknown Reaction Type')

    def build_odes(self):
        """
        Builds rate equations, iterates over states to build ODE
        equations. Casadi model is generated with a call to build_model
        """

        # Clear entries from a previous build
        self.params.reset()
        while self.odes: self.odes.pop()


        # Dictionaries to keep track of where states are
        # produced/consumed
        productdict = {}
        reactantdict = {}
        for state in self.states:
            productdict[str(state)] = []
            reactantdict[str(state)] = []

        for rxn in self.reactions:
            rxn.find_rate()
            # Build dicts
            for reactant in rxn.reactants:
                reactantdict[str(reactant)] += [rxn]
            for product in rxn.products:
                productdict[str(product)] += [rxn]

        for state in self.states:
            temprate = 0
            for rxn in productdict[str(state)]:
                temprate += rxn.rate
            for rxn in reactantdict[str(state)]:
                temprate -= rxn.rate

            self.odes += [temprate]

    def set_parameter(self, parameter_str, val):
        """
        Remove a parameter (parameter_str, string) from the equations by
        setting it to a fixed value (val, float), then removing it from
        the self.params list. Must be called after build_odes and before
        build_model.
        """

        parameter, index = self.params.lookup(parameter_str)
        self.params.pop(index)

        for i, ode in enumerate(self.odes):
            self.odes[i] = cs.substitute(ode, parameter, cs.ssym(val))



    def build_model(self):
        """
        Takes inputs from self.states, self.params and self.odes to
        build a casadi SXFunction model in self.model. Also calculates
        self.NP and self.NEQ
        """
        
        x = cs.vertcat(self.states)
        p = cs.vertcat(self.params)
        ode = cs.vertcat(self.odes)

        t = cs.ssym('t')
        fn = cs.SXFunction(cs.daeIn(t=t, x=x, p=p),
                           cs.daeOut(ode=ode))

        self.model = fn

        self.NP = len(self.params)
        self.NEQ = len(self.states)


    def remove_state(self, state_str):
        """
        Takes a string representation of a model state, removes the
        state from any term in which it might appear
        """

        locations = ['reactants', 'products', 'activators',
                     'inhibitors']

        # Iterate over all reactions, types, and entries to remove the
        # chosen state.
        for reaction in self.reactions:
            for location in locations:
                species = getattr(reaction, location)
                cleaned = []
                for specie in species:
                    parsed = parse_string(specie)
                    try: parsed.remove(state_str)
                    except ValueError: pass
                    cleaned += unparse_string(parsed)
                setattr(reaction, location, cleaned)

        # remove the state from the states list
        state_sx, index = self.states.lookup(state_str)
        self.states.pop(index)

        # remove reactions that have neither reactants or products
        # Identify Reactions
        empty_reactions = []
        for i, reaction in enumerate(self.reactions):
            if not reaction.reactants + reaction.products:
                empty_reactions += [i]

        # Remove identified reactions
        count = 0
        for index in empty_reactions:
            self.reactions.pop(index - count)
            count += 1

    def transcription_function(self, weight):
        """
        Returns a function that can be used with Collocation's
        'minimize_f', such that total transcription is minimized with
        constant weight 'weight'
        """

        x = cs.vertcat(self.states)
        p = cs.vertcat(self.params)

        txn = [rxn.rate for rxn in self.reactions if type(rxn) is
               TranscriptionRxn]

        txn = weight*cs.sumAll(cs.vertcat(txn))

        t = cs.ssym('t')
        fn = cs.SXFunction([x,p],[txn])
        fn.init()

        def return_func(x, p):
            """ Function to return for Collocation """

            try:
                fn.setInput(x, 0)
                fn.setInput(p, 1)
                fn.evaluate()
                return float(fn.output().toArray())
            except Exception:
                return fn.call([x,p])[0]

        return return_func




    def latex_rep(self):
        """
        returns a latex-ready representation of the model equations.
        """

        from CommonFiles.symbolics import LatexVisitor
        import ast

        class ModelLatexVisitor(LatexVisitor):
            """ class to convert strings to latex strings """
            # def __init__(self, states, params):
            #     super(ModelLatexVisitor, self).__init__()
            #     self.model_states = states
            #     self.model_params = params

            def visit_Name(self, n):
                if   n.id in self.model_states.listrep():
                    return r'\mathrm{\bf ' + n.id + r'}'

                elif n.id in self.model_params.listrep():
                    baseindex = n.id.find('_')
                    base = n.id[:baseindex]
                    # Name or index if no name
                    tempname = n.id[baseindex+1:]
                    if '_' in tempname:
                        # name and index
                        ind = tempname.find('_')
                        name = tempname[:ind]
                        pindex = tempname[ind+1:]
                    else:
                        name = tempname
                        pindex = None

                    if pindex: return r'\mathrm{\bf ' + base + r'_' + r'{'\
                       + name + r',' + pindex + r'}' + r'}'
                    else: return r'\mathrm{\bf ' + base + r'_' + r'{'\
                            + name + r'}' + r'}'

                else: return n.id

        visitor = ModelLatexVisitor()
        visitor.model_states = self.states
        visitor.model_params = self.params

        strlist = []
        for i, ode in enumerate(self.odes):
            pt = ast.parse(str(ode))
            lhs = (r'\frac{d\mathrm{\bf ' +
                   self.states.listrep()[i] + r'}}{dt} &= ')

            strlist += [lhs + visitor.visit(pt.body[0].value) + r' \\']

        strlist[-1] = strlist[-1][:-2]

        return strlist


if __name__ == '__main__':
    A = cs.SX('A')
    B = cs.SX('B')
    C = cs.SX('C')
    D = cs.SX('D')

    states = States([A, B, C, D])
    params = Parameters()

    rxns = [
        'A + B --/(C)--> D',
        '~~(A)/(B + D, C)~~> D' ]

    genrxn = GenericRxn(states, params, rxns[0])
    txnrxn = TranscriptionRxn(states, params, rxns[1])

    print 'genrxn test: \t', genrxn.find_rate(), '\t', genrxn
    print 'txnrxn test: \t', txnrxn.find_rate(), '\t', txnrxn

    modeltest = Model(states)
    for rxn in rxns: modeltest.add_reaction(rxn)
    modeltest.build_odes()
    hillparams = modeltest.params.lookup_loose('n')
    # modeltest.set_parameter('n_A', 3)
    modeltest.build_model()
    print modeltest.model.outputSX().toArray()
    rep = modeltest.latex_rep()
    for i in rep:
        print i
