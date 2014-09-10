""" 
    
    StochSS interface to StochKit2.
    
    This module implements a class (StochKitModel) that defines a StochKit2 
    object. StochKitModel extends Model in the 'model.py' section, and supplements 
    it with StochKit2 specific model serialization (to StochKit's naive XML format). 
    
    For examples of use the model API consult the examples in the module 'examplemodels.py'.
    
    The XML serialization is implemented through a StochMLDocument class, which 
    rely on either the lxml or etree modules. To serialize a StochKitModel object 'model',
    simply do
    
        print model.serialize()

    which is equivalent to 
    
        document = StochMLDocument.fromModel(model)
        print document.toString()
    
    You can also initalize a model from an exisiting XML file (See class documentation).
    
    The module also conatains some experimental code for wrapping StochKit output data
    and writing it to .mat files. This should not presently be used by the GAE app.
    
    It also implements a wrapper around StochKit, which uses systems calls to execute
    StochKit and collect its results. This function is mainly inteded to be used by 
    the test suite, but is include here since it can be useful in other contexts as well.
    
    Raises: InvalidModelError, InvalidStochMLError
    
    Andreas Hellander, April 2013.
    
    Minor Changes: J. Abel, April 2014
    
"""
from collections import OrderedDict
import Bioluminescence as bl
import sys

import numpy as np
# import pylab as pl

try:
    import lxml.etree as etree
    no_pretty_print = False
except:
    import xml.etree.ElementTree as etree
    import xml.dom.minidom
    import re
    no_pretty_print = True

try:
    import scipy.io as spio
    isSCIPY = True
except:
    pass

import os
try:
    import shutil
    import numpy
except:
    pass

""" 
    This module describes a model of a well-mixed biochemical system, via the Model class.
    Model objects should not be instantiated by an application. Instead, use StochKitModel 
    in 'stochkit.py', which extends Model with StochKit2 specific serialization. Refer to 
    'stochkit.py' for examples of its use. 
    
    Raises: SpeciesError, ParameterError, ReactionError
    
    Contact: John Abel
    
"""

algorithmlocation = "$HOME/Packages/StochKit2.0.10/ssa"


class Model(object):
    """ Representation of a well mixed biochemical model. Interfaces to solvers in StochSS
        should attempt to extend Model. """
    
    def __init__(self,name="",volume = None):
        """ Create an empty model. """
        
        # The name that the model is referenced by (should be a String)
        self.name = name
        
        # Optional decription of the model (string)
        self.annotation = ""
        
        # Dictionaries with Species, Reactions and Parameter objects.
        # Species,Reactio and Paramter names are used as keys.
        self.listOfParameters = OrderedDict()
        self.listOfSpecies    = OrderedDict()
        self.listOfReactions  = OrderedDict()
        
        # A well mixed model has an optional volume parameter. This should be a Parameter
        self.volume = volume;

        # This defines the unit system at work for all numbers in the model
        #   It should be a logical error to leave this undefined, subclasses should set it
        self.units = None
        
        # Dict that holds flattended parameters and species for
        # evaluation of expressions in the scope of the model.
        self.namespace = OrderedDict([])

    def updateNamespace(self):
        """ Create a dict with flattened parameter and species objects. """
        for param in self.listOfParameters:
            self.namespace[param]=self.listOfParameters[param].value
        # Dictionary of expressions that can be evaluated in the scope of this model.
        self.expressions = {}

    def getSpecies(self, sname):
        return self.listOfSpecies[sname]
    
    def getAllSpecies(self):
        return self.listOfSpecies

    def addSpecies(self, obj):
        """ 
            Add a species to listOfSpecies. Accepts input either as a single Species object, or
            as a list of Species objects.
        """
                
        if isinstance(obj, Species):
            if obj.name in self.listOfSpecies:
                raise ModelError("Can't add species. A species with that name alredy exisits.")
            self.listOfSpecies[obj.name] = obj;
        else: # obj is a list of species
            for S in obj:
                if S.name in self.listOfSpecies:
                    raise ModelError("Can't add species. A species with that name alredy exisits.")
                self.listOfSpecies[S.name] = S;
    
    def deleteSpecies(self, obj):
        self.listOfSpecies.pop(obj)        
         
    def deleteAllSpecies(self):
        self.listOfSpecies.clear()

    def setUnits(self, units):
        if units.lower() == 'concentration' or units.lower() == 'population':
            self.units = units.lower()
        else:
            raise Exception("units must be either concentration or population (case insensitive)")

    def getParameter(self,pname):
        try:
            return self.listOfParameters[pname]
        except:
            raise ModelError("No parameter named "+pname)
    def getAllParameters(self):
        return self.listOfParameters
    
    def addParameter(self,params):
        """ 
            Add Paramter(s) to listOfParamters. Input can be either a single
            paramter object or a list of Parameters.
        """
        # TODO, make sure that you don't overwrite an existing parameter??
        if type(params).__name__=='list':
            for p in params:
                self.listOfParameters[p.name] = p
        else:
            if type(params).__name__=='instance':
                self.listOfParameters[params.name] = params
            else:
                raise

    def deleteParameter(self, obj):
        self.listOfParameters.pop(obj)

    def setParameter(self,pname,expression):
        """ Set the expression of an existing paramter. """
        p = self.listOfParameters[pname]
        p.expression = expression
        p.evaluate()
        
    def resolveParameters(self):
        """ Attempt to resolve all parameter expressions to scalar floats. This
            methods must be called before exporting the model. """
        self.updateNamespace()
        for param in self.listOfParameters:
            try:
                self.listOfParameters[param].evaluate(self.namespace)
            except:
                raise ParameterError("Could not resolve Parameter expression "+param + "to a scalar value.")
    
    def deleteAllParameters(self):
        self.listOfParameters.clear()

    def addReaction(self,reacs):
        """ Add reactions to model. Input can be single instance, a list of instances
            or a dict with name,instance pairs. """
        
        # TODO, make sure that you cannot overwrite an existing parameter
        param_type = type(reacs).__name__
        if param_type == 'list':
            for r in reacs:
                self.listOfReactions[r.name] = r
        elif param_type == 'dict' or param_type == 'OrderedDict':
            self.listOfReactions = reacs
        elif param_type == 'instance':
                self.listOfReactions[reacs.name] = reacs
        else:
            raise

    def getReaction(self, rname):
        return self.listOfreactions[rname]

    def getAllReactions(self):
        return self.listOfReactions
    
    def deleteReaction(self, obj):
        self.listOfReactions.pop(obj)
        
    def deleteAllReactions(self):
        self.listOfReactions.clear()

    def _cmp_(self,other):
        """ Compare """

    

class Species():
    """ Chemical species. """
    
    def __init__(self,name="",initial_value=0):
        # A species has a name (string) and an initial value (positive integer)
        self.name = name
        self.initial_value = initial_value
        assert self.initial_value >= 0, "A species initial value has to be a positive number."

            #def __eq__(self,other):
#  return self.__dict__ == other.__dict__

class Parameter():
    """ 
        A parameter can be given as an expression (function) or directly as a value (scalar).
        If given an expression, it should be understood as evaluable in the namespace
        of a parent Model.
    """
    # AH: Should the parameter, being evaluable, be implemented as a Functor object?

    def __init__(self,name="",expression=None,value=None):

        self.name = name        
        # We allow expression to be passed in as a non-string type. Invalid strings
        # will be caught below. It is perfectly fine to give a scalar value as the expression.
        # This can then be evaluated in an empty namespace to the scalar value.
        self.expression = expression
        if expression != None:
            self.expression = str(expression)
        
        self.value = value
            
        # self.value is allowed to be None, but not self.expression. self.value
        # might not be evaluable in the namespace of this parameter, but defined
        # in the context of a model or reaction.
        if self.expression == None:
            raise TypeError
    
        if self.value == None:
            self.evaluate()
    
    def evaluate(self,namespace={}):
        """ Evaluate the expression and return the (scalar) value """
        try:
            self.value = (float(eval(self.expression, namespace)))
        except:
            self.value = None
            
    def setExpression(self,expression):
        self.expression = expression
        # We allow expression to be passed in as a non-string type. Invalid strings
        # will be caught below. It is perfectly fine to give a scalar value as the expression.
        # This can then be evaluated in an empty namespace to the scalar value.
        if expression != None:
            self.expression = str(expression)
                    
        if self.expression == None:
            raise TypeError
    
        self.evaluate()

class Reaction():
    """ 
        Models a reaction. A reaction has its own dictinaries of species (reactants and products) and parameters.
        The reaction's propensity function needs to be evaluable (and result in a non-negative scalar value)
        in the namespace defined by the union of those dicts.
    """

    def __init__(self, name = "", reactants = {}, products = {}, propensity_function = None, massaction = False, rate=None, annotation=None):
        """ 
            Initializes the reaction using short-hand notation. 
            
            Input: 
                name:                       string that the model is referenced by
                parameters:                 a list of parameter instances
                propensity_function:         string with the expression for the reaction's propensity
                reactants:                  List of (species,stoiciometry) tuples
                product:                    List of (species,stoiciometry) tuples
                annotation:                 Description of the reaction (meta)
            
                massaction True,{False}     is the reaction of mass action type or not?
                rate                        if mass action, rate is a paramter instance.
            
            Raises: ReactionError
            
        """
            
        # Metadata
        self.name = name
        self.annotation = ""
        
        # We might use this flag in the future to automatically generate
        # the propensity function if set to True. 
        self.massaction = massaction

        self.propensity_function = propensity_function
        if self.propensity_function !=None and self.massaction:
            errmsg = "Reaction "+self.name +" You cannot set the propensity type to mass-action and simultaneously set a propensity function."
            raise ReactionError(errmsg)
        
        self.reactants = {}
        for r in reactants:
            rtype = type(r).__name__
            if rtype=='instance':
                self.reactants[r.name] = reactants[r]
            else:
                self.reactants[r]=reactants[r]
    
        self.products = {}
        for p in products:
            rtype = type(p).__name__
            if rtype=='instance':
                self.products[p.name] = products[p]
            else:
                self.products[p]=products[p]

        if self.massaction:
            self.type = "mass-action"
            if rate == None:
                raise ReactionError("Reaction : A mass-action propensity has to have a rate.")
            self.marate = rate
            self.createMassAction()
        else:
            self.type = "customized"
                
    def createMassAction(self):
        """ 
            Create a mass action propensity function given
            self.reactants and a single parameter value.
        """
        # We support zeroth, first and second order propensities only.
        # There is no theoretical justification for higher order propensities.
        # Users can still create such propensities if they really want to,
        # but should then use a custom propensity.
        total_stoch=0
        for r in self.reactants:
            total_stoch+=self.reactants[r]
        if total_stoch>2:
            raise ReactionError("Reaction: A mass-action reaction cannot involve more than two of one species or one of two species.")
        # Case EmptySet -> Y
        propensity_function = self.marate.name;
             
        # There are only three ways to get 'total_stoch==2':
        for r in self.reactants:
            # Case 1: 2X -> Y
            if self.reactants[r] == 2:
                propensity_function = "0.5*" +propensity_function+ "*"+r+"*("+r+"-1)"
            else:
            # Case 3: X1, X2 -> Y;
                propensity_function += "*"+r

        self.propensity_function = propensity_function
            
    def setType(self,type):
        if type.lower() not in {'mass-action','customized'}:
            raise ReactionError("Invalid reaction type.")
        self.type = type.lower()

        self.massaction = False if self.type == 'customized' else True
    
    def addReactant(self,S,stoichiometry):
        if stoichiometry <= 0:
            raise ReactionError("Reaction Stoichiometry must be a positive integer.")
        self.reactants[S.name]=stoichiometry

    def addProduct(self,S,stoichiometry):
        self.products[S.name]=stoichiometry

    def Annotate(self,annotation):
        self.annotation = annotation

# Module exceptions
class ModelError(Exception):
    pass

class SpeciesError(ModelError):
    pass

class ReactionError(ModelError):
    pass

class ParameterError(ModelError):
    pass




class StochKitModel(Model):
    """ StochKitModel extends a well mixed model with StochKit specific serialization. """
    def __init__(self, *args, **kwargs):
        super(StochKitModel, self).__init__(*args, **kwargs)

        self.units = "population"

    def serialize(self):
        """ Serializes a Model object to valid StochML. """
        self.resolveParameters()
        doc = StochMLDocument().fromModel(self)
        return doc.toString()

class StochMLDocument():
    """ Serializiation and deserialization of a StochKitModel to/from 
        the native StochKit2 XML format. """
    
    def __init__(self):
        # The root element
        self.document = etree.Element("Model")
    
    @classmethod
    def fromModel(cls,model):
        """ Creates an StochKit XML document from an exisiting StochKitModel object.
            This method assumes that all the parameters in the model are already resolved
            to scalar floats (see Model.resolveParamters). 
                
            Note, this method is intended to be used interanally by the models 'serialization' 
            function, which performs additional operations and tests on the model prior to 
            writing out the XML file. You should NOT do 
            
            document = StochMLDocument.fromModel(model)
            print document.toString()
            
            you SHOULD do
            
            print model.serialize()            
            
        """
        
        # Description
        md = cls()
        
        d = etree.Element('Description') 

        #
        if model.units.lower() == "concentration":
            d.set('units', model.units.lower())

        d.text = model.annotation
        md.document.append(d)
        
        # Number of Reactions
        nr = etree.Element('NumberOfReactions')
        nr.text = str(len(model.listOfReactions))
        md.document.append(nr)
        
        # Number of Species
        ns = etree.Element('NumberOfSpecies')
        ns.text = str(len(model.listOfSpecies))
        md.document.append(ns)
        
        # Species
        spec = etree.Element('SpeciesList')
        for sname in model.listOfSpecies:
            spec.append(md.SpeciestoElement(model.listOfSpecies[sname]))
        md.document.append(spec)
                
        # Parameters
        params = etree.Element('ParametersList')
        for pname in model.listOfParameters:
            params.append(md.ParametertoElement(model.listOfParameters[pname]))

        #if model.volume != None and model.units == "population":
        #    params.append(md.ParametertoElement(model.volume))

        md.document.append(params)
        
        # Reactions
        reacs = etree.Element('ReactionsList')
        for rname in model.listOfReactions:
            reacs.append(md.ReactionToElement(model.listOfReactions[rname]))
        md.document.append(reacs)
        
        return md
    
    
    @classmethod
    def fromFile(cls,filepath):
        """ Intializes the document from an exisiting native StochKit XML file read from disk. """
        tree = etree.parse(filepath)
        root = tree.getroot()
        md = cls()
        md.document = root
        return md

    @classmethod
    def fromString(cls,string):
        """ Intializes the document from an exisiting native StochKit XML file read from disk. """
        root = etree.fromstring(string)
        
        md = cls()
        md.document = root
        return md

    def toModel(self,name):
        """ Instantiates a StochKitModel object from a StochMLDocument. """
        
        # Empty model
        model = StochKitModel(name=name)
        root = self.document
        
        # Try to set name from document
        if model.name is "":
            name = root.find('Name')
            if name.text is None:
                raise
            else:
                model.name = name.text
        
        # Set annotiation
        ann = root.find('Description')
        if ann is not None:
            units = ann.get('units')

            if units:
                units = units.strip().lower()

            if units == "concentration":
                model.units = "concentration"
            elif units == "population":
                model.units = "population"
            else: # Default 
                model.units = "population"

            if ann.text is None:
                model.annotation = ""
            else:
                model.annotation = ann.text

        # Set units
        units = root.find('Units')
        if units is not None:
            if units.text.strip().lower() == "concentration":
                model.units = "concentration"
            elif units.text.strip().lower() == "population":
                model.units = "population"
            else: # Default 
                model.units = "population"
    
        # Create parameters
        for px in root.iter('Parameter'):
            name = px.find('Id').text
            expr = px.find('Expression').text
            if name.lower() == 'volume':
                model.volume = Parameter(name, expression = expr)
            else:
                p = Parameter(name,expression=expr)
                # Try to evaluate the expression in the empty namespace (if the expr is a scalar value)
                p.evaluate()
                model.addParameter(p)
        
        # Create species
        for spec in root.iter('Species'):
            name = spec.find('Id').text
            val  = spec.find('InitialPopulation').text
            s = Species(name,initial_value = float(val))
            model.addSpecies([s])
        
        # The namespace_propensity for evaluating the propensity function for reactions must
        # contain all the species and parameters.
        namespace_propensity = OrderedDict()
        all_species = model.getAllSpecies()
        all_parameters = model.getAllParameters()
        
        for param in all_species:
            namespace_propensity[param] = all_species[param].initial_value
        
        for param in all_parameters:
            namespace_propensity[param] = all_parameters[param].value
        
        # Create reactions
        for reac in root.iter('Reaction'):
            try:
                name = reac.find('Id').text
            except:
                raise InvalidStochMLError("Reaction has no name.")
            
            reaction  = Reaction(name=name,reactants={},products={})
                
            # Type may be 'mass-action','customized'
            try:
                type = reac.find('Type').text
            except:
                raise InvalidStochMLError("No reaction type specified.")
                    
            reactants  = reac.find('Reactants')
            try:
                for ss in reactants.iter('SpeciesReference'):
                    specname = ss.get('id')
                    # The stochiometry should be an integer value, but some
                    # exising StoxhKit models have them as floats. This is why we
                    # need the slightly odd conversion below. 
                    stoch = int(float(ss.get('stoichiometry')))
                    # Select a reference to species with name specname
                    sref = model.listOfSpecies[specname]
                    try:
                        # The sref list should only contain one element if the XML file is valid.
                        reaction.reactants[specname] = stoch
                    except Exception,e:
                        StochMLImportError(e)
            except:
                # Yes, this is correct. 'reactants' can be None
                pass

            products  = reac.find('Products')
            try:
                for ss in products.iter('SpeciesReference'):
                    specname = ss.get('id')
                    stoch = int(float(ss.get('stoichiometry')))
                    sref = model.listOfSpecies[specname]
                    try:
                        # The sref list should only contain one element if the XML file is valid.
                        reaction.products[specname] = stoch
                    except Exception,e:
                        raise StochMLImportError(e)
            except:
                # Yes, this is correct. 'products' can be None
                pass
                            
            if type == 'mass-action':
                reaction.massaction = True
                reaction.type = 'mass-action'
                # If it is mass-action, a parameter reference is needed.
                # This has to be a reference to a species instance. We explicitly
                # disallow a scalar value to be passed as the paramtete.  
                try:
                    ratename=reac.find('Rate').text
                    try:
                        reaction.marate = model.listOfParameters[ratename]
                    except KeyError, k:
                        # No paramter name is given. This is a valid use case in StochKit.
                        # We generate a name for the paramter, and create a new parameter instance.
                        # The parameter's value should now be found in 'ratename'.
                        generated_rate_name = "Reaction_" + name + "_rate_constant";
                        p = Parameter(name=generated_rate_name, expression=ratename);
                        # Try to evaluate the parameter to set its value
                        p.evaluate()
                        model.addParameter(p)
                        reaction.marate = model.listOfParameters[generated_rate_name]

                    reaction.createMassAction()
                except Exception, e:
                    raise
            elif type == 'customized':
                try:
                    propfunc = reac.find('PropensityFunction').text
                except Exception,e:
                    raise InvalidStochMLError("Found a customized propensity function, but no expression was given."+e)
                reaction.propensity_function = propfunc
            else:
                raise InvalidStochMLError("Unsupported or no reaction type given for reaction" + name)

            model.addReaction(reaction)
        
        return model
    
    def toString(self):
        """ Returns  the document as a string. """
        try:
            return etree.tostring(self.document, pretty_print=True)
        except:
            # Hack to print pretty xml without pretty-print (requires the lxml module).
            doc = etree.tostring(self.document)
            xmldoc = xml.dom.minidom.parseString(doc)
            uglyXml = xmldoc.toprettyxml(indent='  ')
            text_re = re.compile(">\n\s+([^<>\s].*?)\n\s+</", re.DOTALL)
            prettyXml = text_re.sub(">\g<1></", uglyXml)
            return prettyXml
    
    def SpeciestoElement(self,S):
        e = etree.Element('Species')
        idElement = etree.Element('Id')
        idElement.text = S.name
        e.append(idElement)
        
        if hasattr(S, 'description'):
            descriptionElement = etree.Element('Description')
            descriptionElement.text = S.description
            e.append(descriptionElement)
        
        initialPopulationElement = etree.Element('InitialPopulation')
        initialPopulationElement.text = str(S.initial_value)
        e.append(initialPopulationElement)
        
        return e
    
    def ParametertoElement(self,P):
        e = etree.Element('Parameter')
        idElement = etree.Element('Id')
        idElement.text = P.name
        e.append(idElement)
        expressionElement = etree.Element('Expression')
        expressionElement.text = str(P.value)
        e.append(expressionElement)
        return e
    
    def ReactionToElement(self,R):
        e = etree.Element('Reaction')
        
        idElement = etree.Element('Id')
        idElement.text = R.name
        e.append(idElement)
        
        try:
            descriptionElement = etree.Element('Description')
            descriptionElement.text = self.annotation
            e.append(descriptionElement)
        except:
            pass
        
        try:
            typeElement = etree.Element('Type')
            typeElement.text = R.type
            e.append(typeElement)
        except:
            pass
    
        # StochKit2 wants a rate for mass-action propensites
        if R.massaction:
            try:
                rateElement = etree.Element('Rate')
                # A mass-action reactions should only have one parameter
                rateElement.text = R.marate.name
                e.append(rateElement)
            except:
                pass

        else:
            #try:
            functionElement = etree.Element('PropensityFunction')
            functionElement.text = R.propensity_function
            e.append(functionElement)
            #except:
            #    pass

        reactants = etree.Element('Reactants')

        for reactant, stoichiometry in R.reactants.items():
            srElement = etree.Element('SpeciesReference')
            srElement.set('id', reactant)
            srElement.set('stoichiometry', str(stoichiometry))
            reactants.append(srElement)

        e.append(reactants)

        products = etree.Element('Products')
        for product, stoichiometry in R.products.items():
            srElement = etree.Element('SpeciesReference')
            srElement.set('id', product)
            srElement.set('stoichiometry', str(stoichiometry))
            products.append(srElement)
        e.append(products)

        return e


class StochKitTrajectory():
    """
        A StochKitTrajectory is a numpy ndarray.
        The first column is the time points in the timeseries,
        followed by species copy numbers.
    """
    
    def __init__(self,data=None,id=None):
        
        # String identifier
        self.id = id
    
        # Matrix with copy number data.
        self.data = data
        [self.tlen,self.ns] = np.shape(data);

class StochKitEnsemble():
    """ 
        A stochKit ensemble is a collection of StochKitTrajectories,
        all sharing a common set of metadata (generated from the same model instance).
    """
    
    def __init__(self,id=None,trajectories=None,parentmodel=None):
        # String identifier
        self.id = id;
        # Trajectory data
        self.trajectories = trajectories
        # Metadata
        self.parentmodel = parentmodel
        dims = np.shape(self.trajectories)
        self.number_of_trajectories = dims[0]
        self.tlen = dims[1]
        self.number_of_species = dims[2]
    
    def addTrajectory(self,trajectory):
        self.trajectories.append(trajectory)
    
    def dump(self, filename, type="mat"):
        """ 
            Serialize to a binary data file in a matrix format.
            Supported formats are HDF5 (requires h5py), .MAT (for Matlab V. <= 7.2, requires SciPy). 
            Matlab V > 7.3 uses HDF5 as it's base format for .mat files. 
        """
        
        if type == "mat":
            # Write to Matlab format.
            filename = filename
            # Build a struct that contains some metadata and the trajectories
            ensemble = {'trajectories':self.trajectories,'species_names':self.parentmodel.listOfSpecies,'model_parameters':self.parentmodel.listOfParameters,'number_of_species':self.number_of_species,'number_of_trajectories':self.number_of_trajectories}
            spio.savemat(filename,{self.id:ensemble},oned_as="column")
        elif type == "hdf5":
            print "Not supported yet."

class StochKitOutputCollection():
    """ 
        A collection of StochKit Ensembles, not necessarily generated
        from a common model instance (i.e. they do not necessarly have the same metadata).
        This datastructure can be useful to store e.g. data from parameter sweeps, 
        or simply an ensemble of ensembles.
        
        AH: Something like a PyTables object would be very useful here, if working
        in a Python environment. 
        
    """

    def __init__(self,collection=[]):
        self.collection = collection

    def addEnsemble(self,ensemble):
        self.collection.append(ensemble)


def stochkit(model, job_id="",t=20,number_of_trajectories=10,increment=0.01,seed=None,
             algorithm=algorithmlocation):
    """ Call out and run StochKit. Collect the results. This routine is mainly
        intended to be used by the (command line) test suite. 
        
        JHA Edit 4-2014 to remove writing the model into the output folder
        
    """
    # We write all StochKit input and output files to a temporary folder
    prefix_outdir = os.getcwd()+'/stochkit_output'

    # If the base output directory does not exist, we create it
    process = os.popen('mkdir -p ' + prefix_outdir);
    process.close()
    

    # Write a temporary StochKit2 input file.
    if isinstance(model,StochKitModel):
        outfile =  (prefix_outdir + "/stochkit_temp_input_" + str(job_id)
                    + ".xml")
        mfhandle = open(outfile,'w')
        document = StochMLDocument.fromModel(model)

    # If the model is a StochKitModel instance, we serialize it to XML,
    # and if it is an XML file, we just make a copy.
    if isinstance(model,StochKitModel):
        document = model.serialize()
        mfhandle.write(document)
        mfhandle.close()
    elif isinstance(model,str):
        outfile = model

    # Assemble argument list
    ensemblename = job_id
    # If the temporary folder we need to create to hold the output data
    # already exists, we error
    process = os.popen('ls '+prefix_outdir)
    directories = process.read();
    process.close()
    
    outdir = prefix_outdir+'/'+ensemblename
        
    realizations = number_of_trajectories
    if increment == None:
        increment = t/10;

    if seed == None:
        seed = 0

    # Algorithm, SSA or Tau-leaping?
    executable = algorithm
    
    # Assemble the argument list
    args = ''
    args+='--model '
    args+=outfile
    args+=' --out-dir '+outdir
    args+=' -t '
    args+=str(t)
    num_output_points = str(int(float(t/increment)))
    args+=' -i ' + num_output_points
    args+=' --realizations '
    args+=str(realizations)
    if ensemblename in directories:
        print 'Ensemble already existed, using --force.'
        args+=' --force'
    
    # Only use on processor per StochKit job. 
    args+= ' -p 1'
  
    # We keep all the trajectories by default.
    args+=' --keep-trajectories'

    # TODO: We need a robust way to pick a default seed for the ensemble. It needs to be robust in a ditributed, parallel env.
    args+=' --seed '
    args+=str(seed)

    # If we are using local mode, shell out and run StochKit (SSA or Tau-leaping)
    cmd = executable+' '+args

    # Can't test for failed execution here, popen does not return stderr.
    process = os.popen(cmd)

    stochkit_output_message = process.read()
    process.close()
    

    # Collect all the output data
    files = os.listdir(outdir + '/stats')
       
    trajectories = []
    files = os.listdir(outdir + '/trajectories')
    

    for filename in files:
        if 'trajectory' in filename:
            trajectories.append(numpy.loadtxt(outdir + '/trajectories/'
                                              + filename))
        else:
            sys.stderr.write('Couldn\'t identify file (' + filename + ') found in output folder')
            sys.exit(-1)

    # Clean up
    shutil.rmtree(outdir)

    return trajectories

# Exceptions
class StochMLImportError(Exception):
    pass

class InvalidStochMLError(Exception):
    pass




#========================================================================
# Stochastic Evaluation
#========================================================================

class StochEval(object):
    """
    For the evaluation of Stochastic output data from the SSA.
    """
    def __init__(self, trajectories,state_names,param_names,vol,timeshift=0):
        
        #Determines the size of data we are working with.
        self.trajectories = trajectories
        self.state_names  = state_names
        self.param_names  = param_names
        self.trajcount    = len(trajectories[:])
        self.timecount    = len(trajectories[0][:,0])
        self.time         = trajectories[0][:,0]+timeshift
        self.statecount   = len(trajectories[0][0,:])
        self.vol          = vol
        
        #Creates dictionaries for state indexes
        self.ydict = {}
        for par,ind in zip(state_names,range(0,self.statecount)):
            self.ydict[par] = ind+1 #The +1 is because the first is actually time
        
        A = np.zeros((self.trajcount,self.timecount,self.statecount))
        #Creates 3D array of trajectories
        for i in range(self.trajcount):
            A[i,:,:]=self.trajectories[i]
        self.avgtraj = np.mean(A,axis=0)
    
    def bl_obj(self,SV='p'):
        #Initializes a Bioluminescence object
        statevarnum = self.ydict[SV]
        bl_obj = bl.Bioluminescence(self.avgtraj[:,0],self.avgtraj[:,statevarnum])
        return bl_obj
        
    def avgtraj(self):
        return self.avgtraj
    
    # def PlotAvg(self,SV,traces=True,fignum=1,color='black',conc=False):
    #     
    #     avg=self.avgtraj
    #     pl.figure(fignum)
    #     statevarnum = self.ydict[SV]
    #     
    #     if traces is True:
    #         for i in range(self.trajcount):
    #             trj=self.trajectories[i][:,statevarnum]
    #             if conc==True:
    #                 trj=self.trajectories[i][:,statevarnum]/self.vol
    #             pl.plot(self.time,trj,color="gray",linewidth=0.5)
    #     if conc==True:
    #         avg[:,statevarnum]=avg[:,statevarnum]/self.vol
    #     pl.plot(self.time,avg[:,statevarnum],color=color,linewidth=1.5,label = SV)
    #     pl.title('State Variable Ocsillation')
    #     pl.xlabel('Time, Circadian Hours')
    #     pl.ylabel('State Variable')
    #     pl.legend()
    
    def SSAperiod(self,SV='p'):
        #Uses a FFT method to quickly find the most likely period.
        statevarnum = self.ydict[SV]
        print bl.estimate_period(self.time,self.avgtraj[:,statevarnum])
    
    # def waveletplot(self,bl_obj,fignum=10,subplot=111):
    #     """
    #     Uses Bioluminescence module to make a wavelet plot
    #     """
    #     bl_obj.dwt_breakdown()
    #     fig = pl.figure(fignum)
    #     ax = fig.add_subplot(subplot)
    #     bl_obj.plot_dwt_components(ax)
        
    # def lombscargle(self,SV='p',fignum=12):
    #     pl.figure(fignum)
    #     statevarnum = self.ydict[SV]
    #     a0=bl.periodogram(self.time,self.avgtraj[:,statevarnum])[0]
    #     a1=bl.periodogram(self.time,self.avgtraj[:,statevarnum])[1]
    #     pl.plot(a0,a1)
    #     pl.xlabel('Period')
    
    # def decay_fit(self,bl_obj,weights=None,fignum=14):
    #     #Compares fit by decaying sinusoid model with the SSA results
    #     #should not run this for stoch model, detrending not necessary. to detrend, you should 
    #     # instead just subtract the mean from the deterministic model version...
    #     bl_obj.detrend()
    #     #print 'Note: detrending not fixed for model version yet'
    #     bl_obj.fit_sinusoid()
    #     
    #     #Plots original with detrended model
    #     fig = pl.figure(fignum)
    #     ax = fig.add_subplot(111)

    #     ax.plot(bl_obj.x, bl_obj.y)
    #     ax.plot(bl_obj.x, bl_obj.yvals['model'], '--')
    #     ax.plot(bl_obj.x, bl_obj.yvals['mean'],'r--')
    
    # def cwt_plot(self,bl_obj,fignum=13):
    #     """plots the continuous wavelet transform"""
    #     bl_obj.continuous_wavelet_transform(edge_method='exp_sin')
    #     fig = pl.figure()
    #     ax = fig.add_subplot(111)
    #     cme = ax.pcolormesh(bl_obj.cwt['x'], bl_obj.cwt['tau'], bl_obj.cwt['cwt_abs'])
    #     ax.plot(bl_obj.cwt['x'], bl_obj.cwt['period'], 'k')
    #     cme.set_rasterized(True)
    #     ax.set_xlim([bl_obj.cwt['x'].min(), bl_obj.cwt['x'].max()])
    #     ax.set_ylim([bl_obj.cwt['tau'].min(), bl_obj.cwt['tau'].max()])
    #     fig.tight_layout(pad=0.05,h_pad = 0.6,w_pad = 0.6)



class StochPopEval(object):
    """
    For the evaluation of Stochastic output data from the SSA.
    """
    def __init__(self, trajectories,state_names,param_names,vol,EqCount,timeshift=0):
        
        #Determines the size of data we are working with.
        self.trajectories = trajectories
        self.state_names  = state_names
        self.param_names  = param_names
        self.trajcount    = len(trajectories[:])
        self.timecount    = len(trajectories[0][:,0])
        self.time         = trajectories[0][:,0]+timeshift
        self.statecount   = len(trajectories[0][0,:])
        self.vol          = vol
        self.EqCount      = EqCount
        
        #Creates dictionaries for state indexes
        self.ydict = {}
        for par,ind in zip(state_names,range(0,self.statecount)):
            self.ydict[par] = ind+1 #The +1 is because the first is actually time
        
        A = np.zeros((self.trajcount,self.timecount,self.statecount))
        #Creates 3D array of trajectories
        for i in range(self.trajcount):
            A[i,:,:]=self.trajectories[i]
        self.avgtraj = np.mean(A,axis=0)

        

        
    def avgtraj(self):
        return self.avgtraj
        
    def PlotPop(self,SV,fignum=1,color='black',conc=False,traces=True,
                timeshift=0):

        SVind = self.ydict[SV+'_0_0']
        
        SVindexes = np.array([SVind])
        
        for i in range(len(self.state_names)/self.EqCount-1):
            SVind = SVind+self.EqCount
            SVindexes = np.append(SVindexes,SVind)
        
        
        pl.figure(fignum)

        if traces is True:
            for i in range(len(SVindexes)):
                pl.plot(self.time,self.trajectories[0][:,SVindexes[i]],color="gray")
                
        averageSV = np.mean(self.trajectories[0][:,SVindexes],1)
        
        pl.plot(self.time,averageSV,color=color,label = SV)
        pl.title('State Variable Ocsillation')
        pl.xlabel('Time, Circadian Hours')
        pl.ylabel('State Variable')
        pl.legend()

    def PlotPopPartial(self, SV, tstart=0, tend=None, fignum=1,
                       color='black', conc=False, traces=True):
        #Plot for partial time set
        
        if tend == None:
            tend=np.amax(self.time)
        
        end_ind = np.argmin(abs(self.time-tend))
        start_ind = np.argmin(abs(self.time-tstart))
        
        SVind = self.ydict[SV+'_0_0']
        
        SVindexes = np.array([SVind])
        
        for i in range(len(self.state_names)/self.EqCount-1):
            SVind = SVind+self.EqCount
            SVindexes = np.append(SVindexes,SVind)
        
        
        pl.figure(fignum)

        if traces is True:
            for i in range(len(SVindexes)):
                pl.plot(self.time[start_ind:end_ind],self.trajectories[0][start_ind:end_ind,SVindexes[i]],color="gray")
                
        averageSV = np.mean(self.trajectories[0][start_ind:end_ind,SVindexes],1)
        
        pl.plot(self.time[start_ind:end_ind],averageSV,color=color,label = SV)
        pl.title('State Variable Ocsillation')
        pl.xlabel('Time, Circadian Hours')
        pl.ylabel('State Variable')
        pl.legend()
        
        
    def bl_obj(self,SV='p'):
        #Initializes a Bioluminescence object
        
        SVind = self.ydict[SV+'_0_0']
        
        SVindexes = np.array([SVind])
        
        for i in range(len(self.state_names)/self.EqCount-1):
            SVind = SVind+self.EqCount
            SVindexes = np.append(SVindexes,SVind)
            
        averageSV = np.mean(self.trajectories[0][:,SVindexes],1)   

        bl_obj = bl.Bioluminescence(self.avgtraj[:,0],averageSV)
        return bl_obj

    def cwt_plot(self,bl_obj,fignum=13):
        """plots the continuous wavelet transform"""
        bl_obj.continuous_wavelet_transform(edge_method='exp_sin')
        fig = pl.figure()
        ax = fig.add_subplot(111)
        cme = ax.pcolormesh(bl_obj.cwt['x'], bl_obj.cwt['tau'], bl_obj.cwt['cwt_abs'])
        ax.plot(bl_obj.cwt['x'], bl_obj.cwt['period'], 'k')
        cme.set_rasterized(True)
        ax.set_xlim([bl_obj.cwt['x'].min(), bl_obj.cwt['x'].max()])
        ax.set_ylim([bl_obj.cwt['tau'].min(), bl_obj.cwt['tau'].max()])
        fig.tight_layout(pad=0.05,h_pad = 0.6,w_pad = 0.6)

    def waveletplot(self,bl_obj,fignum=10,subplot=111):
        """
        Uses Bioluminescence module to make a wavelet plot
        """
        bl_obj.dwt_breakdown()
        fig = pl.figure(fignum)
        ax = fig.add_subplot(subplot)
        bl_obj.plot_dwt_components(ax)

    def lombscargle(self,SV='p',fignum=12):
        
        SVind = self.ydict[SV+'_0_0']
        
        SVindexes = np.array([SVind])
        
        for i in range(len(self.state_names)/self.EqCount-1):
            SVind = SVind+self.EqCount
            SVindexes = np.append(SVindexes,SVind)
            
        averageSV = np.mean(self.trajectories[0][:,SVindexes],1)
        pl.figure(fignum)

        a0=bl.periodogram(self.time,averageSV)[0]
        a1=bl.periodogram(self.time,averageSV)[1]
        pl.plot(a0,a1)
        pl.xlabel('Period')


