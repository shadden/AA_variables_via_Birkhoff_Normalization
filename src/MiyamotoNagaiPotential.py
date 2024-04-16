import sympy as sp
import celmech as cm

def MiyamotoNagai_get_hamiltonian_full(ics,Na,Nb):
    r"""
    Get Hamiltonian for orbits in a Miyamato-Nagai potential,

    .. math::
        \Phi(R,z) =  -\frac{1}{\sqrt{R^2} + (a + \sqrt{z^2+ b^2})^2}

    Parameters
    ----------
    ics : ndarray
        Array of initial conditions for (L,R,z,phi,p_R,p_z)
    Na : float
        value of the parameter a
    Nb : float
        value of the parameter b

    Returns
    -------
    celmech.hamiltonian.Hamiltonian
        Hamiltonian object
    """
    L,R,z,phi,pR,pz = sp.symbols("L,R,z,phi,p_R,p_z",real=True)
    a,b = sp.symbols("a,b",real=True)
    KE = (pR * pR + pz * pz + (L/R) * (L/R))/2
    z_term = a + sp.sqrt(z*z + b*b)
    PE = -1/sp.sqrt(R * R + z_term * z_term)
    H = KE + PE
    pars = {a:Na,b:Nb}
    state = cm.PhaseSpaceState((phi,R,z,L,pR,pz),ics)
    ham = cm.Hamiltonian(H,pars,state)
    return ham

def MiyamotoNagai_get_hamiltonian(ics,NL,Na,Nb):
    r"""
    Get Hamiltonian for orbits in a Miyamato-Nagai potential,

    .. math::
        \Phi(R,z) =  -\frac{1}{\sqrt{R^2} + (a + \sqrt{z^2+ b^2})^2}

    Parameters
    ----------
    ics : ndarray
        Array of initial conditions for (R,z,p_R,p_z)
    NL : float
        Value of the angular momentum
    Na : float
        value of the parameter a
    Nb : float
        value of the parameter b

    Returns
    -------
    celmech.hamiltonian.Hamiltonian
        Hamiltonian object
    """
    R,z,pR,pz = sp.symbols("R,z,p_R,p_z",real=True)
    a,b,L = sp.symbols("a,b,L",real=True)
    KE = (pR * pR + pz * pz + (L/R) * (L/R))/2
    z_term = a + sp.sqrt(z*z + b*b)
    PE = -1/sp.sqrt(R * R + z_term * z_term)
    H = KE + PE
    pars = {a:Na,b:Nb,L:NL}
    state = cm.PhaseSpaceState((R,z,pR,pz),ics)
    ham = cm.Hamiltonian(H,pars,state)
    return ham
    
from scipy.optimize import root_scalar
def MiyamotoNagai_L_to_Rc(L,a,b):
    """
    Given a value of angular momentum and potential parameters a and b for a
    Miyamoto-Nagai potential, compute the radius of a circular orbit.

    Parameters
    ----------
    L : float
        angular momentum
    a : float
        potential parameter
    b : float
        potential parameter

    Returns
    -------
    _type_
        _description_
    """
    c = a + b
    PhiEff = lambda R: 0.5 * L*L/R/R - 1/jnp.sqrt(R*R + c*c)
    dPhiEff_dR = lambda R: -1 * L*L/R/R/R + R / (R*R + c*c)**(1.5)
    d2PhiEff_d2R = lambda R: 3 * L*L/R/R/R/R + 1 / (R*R + c*c)**(1.5) - 3 * R * R / (R*R + c*c)**(2.5)
    root_result = root_scalar(dPhiEff_dR,x0 = L*L,fprime=d2PhiEff_d2R)
    assert root_result.converged, "Root finding failed!"
    return root_result.root