# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:05:55 2014

@author: john
"""
import stochkit_resources as stk
import numpy as np
import pdb

#
# ========= ODE MODEL TOOLKIT =====================
#

def transcription(rep1,rep2,vmax,km,n):
        return vmax/(km + (rep1 + rep2)**n)
    
def rtranscription(rep1,rep2,vmax,km,n):
    return vmax/(km + (rep1 + rep2)**(n))
    
def ptranscription(pro,vmax,km,n):
    return (vmax*pro**n)/(km+pro**n)
    
def prtranscription(pro,vmax,km,rep1,rep2):
    return (vmax*pro)/((km+pro)*(km+rep1+rep2))

def prtranscription2(pro,rep1,rep2,v1,v2,km):
    return (v1*pro+v2)/(km+rep1+rep2)
def translation(mrna,kt):
    return kt*mrna

def michaelisMenten(s,vmax,km):
    return vmax*s/(km+s)
   
def sharedDegradation(currentspecies,otherspecies,vmax,km):
    return vmax*(currentspecies)/(km + (currentspecies + otherspecies))

def HillTypeActivation(a, vmax, km, n):
    return (a*vmax)/(km + a**n)
    
def HillTypeRepression(rep1,rep2, vmax, km, n):
    return (vmax)/(km + (rep1+rep2)**n)

def Complexing(ka,species1,species2,kd,complex):
    #Leave positive for complexes, negative for reacting species
    return ka*species1*species2 - kd*complex

def lineardeg(k,species):
    return -k*species














#
# ========= SSA MODEL TOOLKIT =====================
#
class SSA_builder(object):
    
    def __init__(self,species_array,param_array,y0in,param,SSAmodel,vol):
        """ Sets up the necessary information for generating the
        reactions"""
        
        self.species_array = list(species_array)
        self.param_array   = list(param_array)
        self.y0in          = np.array(y0in)
        self.param         = np.array(param)
        self.SSAmodel      = SSAmodel
        self.vol           = vol
        
        ParamCount = len(self.param)
        EqCount    = len(self.y0in)
        #Creates dictionaries for state and parameter indexes
        self.pdict = {}
        self.ydict = {}
        
        
        for par,ind in zip(param_array,range(0,ParamCount)):
            self.pdict[par] = ind
        

        for par,ind in zip(species_array,range(0,EqCount)):
            self.ydict[par] = ind
        
        #Creates dictionaries for parameters and state initial values
        self.pvaldict = {}
        self.y0dict = {}
        
        for par,ind in zip(param_array,range(0,ParamCount)):
            self.pvaldict[par] = self.param[ind]
        
        for par,ind in zip(species_array,range(0,EqCount)):
            self.y0dict[par] = self.y0in[ind]

        # Species defined and added to model
        for i in range(len(species_array)):
            self.species_array[i] = stk.Species(name=self.species_array[i],initial_value=self.y0dict[self.species_array[i]])
            self.SSAmodel.addSpecies([self.species_array[i]])
        
        # Parameters defined and added to model
        for i in range(len(param_array)):
            self.param_array[i] = stk.Parameter(name=self.param_array[i],expression=self.pvaldict[self.param_array[i]])
            self.SSAmodel.addParameter([self.param_array[i]])

        self.paramarraycopy = param_array[:]

    #REACTION TYPES =============================
    
    #Mass-Action
    def SSA_MA(self, name='', reactants={}, products={}, rate=''):
        """ Deal with mass-action rates where volume affects the
        stochastic propensity """
        
        # Scale rate by appropriate volume power
        degree = sum(reactants.values())
        if str(self.pvaldict[rate.name]) == rate.expression:
                self.SSAmodel.setParameter(
                    rate.name,
                    (rate.expression +
                    '/pow(' + str(self.vol) + ', ' + str(degree - 1) + ')')
                )

        rxn = stk.Reaction(name=name,
                            reactants=reactants,
                            products=products,
                            massaction=True,
                            rate=rate)

        self.SSAmodel.addReaction(rxn)


    def SSA_MA_tln(self,Desc,P,k,mRNA):
        rxn=stk.Reaction(name=Desc,
                            products={self.species_array[self.ydict[P]]:1},
                            propensity_function=mRNA+"*"+k,
                            annotation="EmptySet->"+P)    
        self.SSAmodel.addReaction(rxn)
        #print Desc+' added successfully.'
        return
        
    def SSA_MA_deg(self,Desc,S,k):
        rxn=stk.Reaction(name=Desc,
                            reactants={self.species_array[self.ydict[S]]:1},
                            propensity_function=k+'*'+S)
        self.SSAmodel.addReaction(rxn)
        #print Desc+' added successfully.'
        return
        
    def SSA_MF_deg(self,Desc,S,k,xlen,ylen):
        cellcount=xlen*ylen
        rxn=stk.Reaction(name=Desc,
                            reactants={self.species_array[self.ydict[S]]:1},
                            propensity_function=k+'*'+S+'/'+str(cellcount))
        self.SSAmodel.addReaction(rxn)
        #print Desc+' added successfully.'
        return
        
    def SSA_MA_complex(self,Desc,sub1,sub2,complex,kf,kr):
        #rescale kf for second order stochastic propensity
        if str(self.pvaldict[kf]) == self.SSAmodel.listOfParameters[kf].expression:
                self.SSAmodel.setParameter(kf,self.SSAmodel.listOfParameters[kf].expression+'/('+str(self.vol)+')')
        rxn1=stk.Reaction(name=Desc+"_formation",
                            reactants={self.species_array[self.ydict[sub1]]:1,
                                       self.species_array[self.ydict[sub2]]:1},
                            products={self.species_array[self.ydict[complex]]:1},
                            massaction=True,
                            rate=self.param_array[self.pdict[kf]])
        self.SSAmodel.addReaction(rxn1)

        rxn2=stk.Reaction(name=Desc+"degradation",
                          reactants={self.species_array[self.ydict[complex]]:1},
                            products={self.species_array[self.ydict[sub1]]:1,
                                       self.species_array[self.ydict[sub2]]:1},
                            massaction=True,
                            rate=self.param_array[self.pdict[kr]])
        self.SSAmodel.addReaction(rxn2)
        
        #print Desc+' added successfully.'
        return
        
    
    def SSA_MA_meanfield(self,sharedspecies,indx,xlen,indy,ylen,kcouple):
        
        #set-up indexes for mixing. Note that this makes sure there is only one reaction each time
        #since the inxed relationship only goes in one direction
        annt = None
        index = '_'+str(indx)+'_'+str(indy)
        
        if indx+1 == xlen: 
            ind_x1 = '_'+str(0)+'_'+str(indy)
        else: 
            ind_x1 = '_'+str(indx+1)+'_'+str(indy)
        
        if indy+1 == ylen: 
            ind_y1 = '_'+str(indx)+'_'+str(0)
        else: 
            ind_y1 = '_'+str(indx)+'_'+str(indy+1)
        
        rctsf={self.species_array[self.ydict[sharedspecies+index]]:1}
        prodsxf={self.species_array[self.ydict[sharedspecies+ind_x1]]:1}
        prodsyf={self.species_array[self.ydict[sharedspecies+ind_y1]]:1}
        
        prodsr = {self.species_array[self.ydict[sharedspecies+index]]:1}
        rctsxr={self.species_array[self.ydict[sharedspecies+ind_x1]]:1}
        rctsyr={self.species_array[self.ydict[sharedspecies+ind_y1]]:1}
        
        propfcnxr = kcouple+'*'+sharedspecies+ind_x1#('+sharedspecies+ind_x1+'-'+sharedspecies+index+')'
        propfcnyr = kcouple+'*'+sharedspecies+ind_y1#('+sharedspecies+ind_y1+'-'+sharedspecies+index+')'
        
        propfcnxf = kcouple+'*'+sharedspecies+index#('+sharedspecies+index+'-'+sharedspecies+ind_x1+')'
        propfcnyf = kcouple+'*'+sharedspecies+index#('+sharedspecies+index+'-'+sharedspecies+ind_y1+')'
        
        
        #Add forward and backward mixing for x and y both
        rxn1 = stk.Reaction(name=sharedspecies+'mixing'+index+'_xf',  
                            reactants=rctsf,
                            products=prodsxf,
                            propensity_function=propfcnxf,
                            annotation=annt)
        self.SSAmodel.addReaction(rxn1)
        
        rxn3 = stk.Reaction(name=sharedspecies+'mixing'+index+'_xr',  
                            reactants=rctsxr,
                            products=prodsr,
                            propensity_function=propfcnxr,
                            annotation=annt)
        self.SSAmodel.addReaction(rxn3)
        
        rxn2 = stk.Reaction(name=sharedspecies+'mixing'+index+'_yf',  
                            reactants=rctsf,
                            products=prodsyf,
                            propensity_function=propfcnyf,
                            annotation=annt)
        self.SSAmodel.addReaction(rxn2)
        
        rxn4 = stk.Reaction(name=sharedspecies+'mixing'+index+'_yr',  
                            reactants=rctsyr,
                            products=prodsr,
                            propensity_function=propfcnyr,
                            annotation=annt)
        self.SSAmodel.addReaction(rxn4)

    #MichaelisMenten Nonlinear (p=P)
    def SSA_MM_C2(self, Desc, vmax, km=None, Rct=None, Prod=None,
                 Act=None, Rep=None, mf=None, M=None):
        """ Peter's hacked function to allow a multiplicitive
        degradation term """

        arg = self.SSA_MM(Desc, vmax, km, Rct, Prod, Act, Rep, mf)
        rxn = self.SSAmodel.listOfReactions[Desc]

        arg['vmax'] += '*' + M

        propfcn = (arg['vmax'] + arg['rctmult'] + arg['actmult'] + '/('
                   + arg['km0'] + arg['rctsum'] + arg['actsum'] +
                   arg['repsum'] + ')')

        rxn.propensity_function = propfcn

        
    #MichaelisMenten Nonlinear (p=P)
    def SSA_MM_P(self, Desc, vmax, km=None, Rct=None, Prod=None,
                 Act=None, Rep=None, mf=None, P=None):
        """ Peter's hacked function to allow for third order hill
        kinetics """

        arg = self.SSA_MM(Desc, vmax, km, Rct, Prod, Act, Rep, mf)
        rxn = self.SSAmodel.listOfReactions[Desc]

        # Add the 3rd order hill term
        arg['repsum'] = (arg['repsum'][0] + 
                         '*'.join(['(' + arg['repsum'][1:] + ')']*P))

        vmax_expr = (str(self.param[self.pdict[arg['vmax']]]) + '*' +
                     str(self.vol) + '**' + str(P+1))
        km0_expr = (str(self.param[self.pdict[arg['km0']]]) + '*' +
                     str(self.vol) + '**' + str(P))

        self.param_array[self.pdict[arg['vmax']]].expression = vmax_expr
        self.param_array[self.pdict[arg['km0']]].expression = km0_expr

        propfcn = (arg['vmax'] + arg['rctmult'] + arg['actmult'] + '/('
                   + arg['km0'] + arg['rctsum'] + arg['actsum'] +
                   arg['repsum'] + ')')

        rxn.propensity_function = propfcn


    #MichaelisMenten Nonlinear
    def SSA_MM(self, Desc, vmax, km=None, Rct=None, Prod=None, Act=None,
               Rep=None, mf=None):
        """ general michaelis-menten term that includes re-scaling for a
        population model """
        
        #counts components
        try: rctcount = len(Rct)
        except: rctcount = 0
        try: prodcount = len(Prod)
        except: prodcount = 0
        try: actcount = len(Act)
        except: actcount = 0
        try: repcount = len(Rep)       
        except: repcount = 0
        
        # Adds Reactants
        rcts = {}
        if Rct is not None:
            rcts.update({self.species_array[self.ydict[Rct[0]]]:1})
        
        # Adds Products
        prods = {}
        for i in range(prodcount):
            prods.update({self.species_array[self.ydict[Prod[i]]]:1})
        
        # redefine vmax -- need some way to determine if it has been
        # updated or not
        if (str(self.pvaldict[vmax]) ==
            self.SSAmodel.listOfParameters[vmax].expression):

            self.SSAmodel.setParameter(
                vmax, self.SSAmodel.listOfParameters[vmax].expression +
                '*(' + str(self.vol) + '**2)/(' + str(self.vol) + '**('
                + str(actcount + (Rct is not None)) + '))')
        
        if len(km)==1:
            # redefine k -- need some way to determine if it has been
            # updated or not
            if (str(self.pvaldict[km[0]]) ==
                self.SSAmodel.listOfParameters[km[0]].expression):

                self.SSAmodel.setParameter(
                    km[0],
                    self.SSAmodel.listOfParameters[km[0]].expression +
                    '*(' + str(self.vol) + ')')
            
            # propensity funtion setup
            rctmult=''
            rctsum =''
            for i in range(Rct is not None):
                rctmult = rctmult + '*' + Rct[i]

            for i in range(rctcount):
                rctsum  = rctsum  + '+' + Rct[i]
                
            actmult = ''
            actsum = ''
            for i in range(actcount):
                actsum  = actsum  + '+' + Act[i]
                actmult = actmult + '*' + Act[i]
            
            repmult = ''
            repsum = ''
            for i in range(repcount):
                repsum  = repsum +'+'+Rep[i]
                repmult = repmult+'*'+Rep[i]
            
            # creates the propensity function
            propfcn = (vmax + rctmult + actmult + '/(' + km[0] + rctsum
                       + actsum + repsum + ')')
            propfcn_args = {
                'vmax'    : vmax,
                'rctmult' : rctmult,
                'actmult' : actmult,
                'km0'     : km[0],
                'rctsum'  : rctsum,
                'actsum'  : actsum,
                'repsum'  : repsum,
            }
            
            # delete this: meanfield approximation only
            if mf is not None:
                xlen,ylen = mf
                cellcount = xlen*ylen
                propfcn = (vmax + rctmult + actmult + '/(' + km[0] +
                           rctsum + actsum + repsum + ')' + '/' +
                           str(cellcount))
                
            annt = 0 # no annotations necessary yet 

            rxn = stk.Reaction(name=Desc, reactants=rcts,
                               products=prods,
                               propensity_function=propfcn,
                               annotation=annt)
        self.SSAmodel.addReaction(rxn)
        #print Desc+' added successfully.'
        return propfcn_args

        if len(km) is not 1:
            print 'I haven\'t made this one yet...'


        def SSA_genMM(self,Desc,vmax,km=None,Sub=None,Prod=None,Act=None,CInh=None,UCInh=None,NCInh=None):
            """general michaelis-menten term that includes re-scaling for a population model
            
                        
            """
            
            #counts components
            try: rctcount = len(Rct)
            except: rctcount = 0
            try: prodcount = len(Prod)
            except: prodcount = 0
            try: actcount = len(Act)
            except: actcount = 0
            try: repcount = len(Rep)       
            except: repcount = 0
            
            #Adds Reactants
            rcts = {}
            if Rct is not None:
                rcts.update({self.species_array[self.ydict[Rct[0]]]:1})
            
            #Adds Products
            prods = {}
            for i in range(prodcount):
                prods.update({self.species_array[self.ydict[Prod[i]]]:1})
            
            #redefine vmax -- need some way to determine if it has been updated or not
            if str(self.pvaldict[vmax]) == self.SSAmodel.listOfParameters[vmax].expression:
                self.SSAmodel.setParameter(vmax,self.SSAmodel.listOfParameters[vmax].expression+'*('+str(self.vol)+'**2)/('
                                        +str(self.vol)+'**('+str(actcount+(Rct is not None))+'))')
            
            if len(km)==1:
                
                #redefine k -- need some way to determine if it has been updated or not
                if str(self.pvaldict[km[0]]) == self.SSAmodel.listOfParameters[km[0]].expression:
                    self.SSAmodel.setParameter(km[0],self.SSAmodel.listOfParameters[km[0]].expression+'*('+str(self.vol)+')')
                
                #propensity funtion setup
                rctmult=''
                rctsum =''
                for i in range(Rct is not None):
                    rctmult = rctmult+'*'+Rct[i]
    
                for i in range(rctcount):
                    rctsum  = rctsum +'+'+Rct[i]
                    
                actmult = ''
                actsum = ''
                for i in range(actcount):
                    actsum  = actsum +'+'+Act[i]
                    actmult = actmult+'*'+Act[i]
                
                repmult = ''
                repsum = ''
                for i in range(repcount):
                    repsum  = repsum +'+'+Rep[i]
                    repmult = repmult+'*'+Rep[i]
                
                #creates the propensity function
                propfcn = vmax+rctmult+actmult+'/('+km[0]+rctsum+actsum+repsum+')'
                
                annt = 0 #no annotations necessary yet 
                
                rxn = stk.Reaction(name=Desc,  
                                reactants=rcts,
                                products=prods,
                                propensity_function=propfcn,
                                annotation=annt)
            self.SSAmodel.addReaction(rxn)
            #print Desc+' added successfully.'
            return


    def SSA_tyson_x(self,Desc,X,Y,P):
        rcts = {}
        prods = {self.species_array[self.ydict[X]]:1}

        exp_P = int(float(self.SSAmodel.listOfParameters[P].expression))

        ymult = Y+('*'+Y)*(exp_P-1)
        volmult = '('+str(self.vol)+('*'+str(self.vol))*(exp_P-1)+')'

        propfcn = str(self.vol)+'*1/(1+('+ymult+'/('+volmult+')))'

        rxn = stk.Reaction(name=Desc,  
                                reactants=rcts,
                                products=prods,
                                propensity_function=propfcn,
                                annotation=None)
        self.SSAmodel.addReaction(rxn)
        
        
        return
    
    def SSA_tyson_y(self,Desc,Y,a0,a1,a2):
        rcts = {self.species_array[self.ydict[Y]]:1}
        prods = {}
        
        propfcn = (Y+'/(a0 + a1*('+Y+'/'+str(self.vol)+
                   ') + a2*'+Y+'*'+Y+'/('+str(self.vol)+'*'
                   +str(self.vol)+'))')

        rxn = stk.Reaction(name=Desc,  
                                reactants=rcts,
                                products=prods,
                                propensity_function=propfcn,
                                annotation=0)
        self.SSAmodel.addReaction(rxn)
        return

    @staticmethod
    def expand_labelarray(labels, nrep):
        """ Function to add indicies to the labels and return a
        flattened matrix such that uncoupled cells from a large
        population can be tracked in a single stochss model """

        # Some string array magic
        nlab = len(labels)
        labels = np.tile(np.asarray(labels), (nrep, 1))
        indicies = np.tile(np.arange(nrep).astype(str), (nlab, 1)).T
        score = np.tile(np.array('_'), (nrep, nlab))

        # Build labels (dont know why element-wise addition is so
        # convoluted)
        # L + _ + i
        out = np.core.defchararray.add(labels, score)
        out = np.core.defchararray.add(out, indicies)
        return out.flatten(order='C')























