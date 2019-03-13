from __future__ import print_function, division

from sympy.core.singleton import S 
from sympy.core import Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma

from sympy.polys.orthopolys import legendre_poly  
from sympy.functions.special.polynomials import legendre # use unedited Legendre polynomial for m=0
_x = Dummy('x')


class assoc_legendre_schmidt_norm(Function):
    r"""
    assoc_legendre_schmidt_norm(n,m, x) gives :math:`S_n^m(x)`, where n and m are
    the degree and order or an expression which is related to the nth
    order Legendre polynomial, :math:`P_n(x)` and the associated non-normalised Legendre
    polynomials :math:`P_n^m(x)`
    in the following manner:

    .. math::
        P_n^m(x) = (1 - x^2)^{\frac{m}{2}}
                   \frac{\mathrm{d}^m P_n(x)}{\mathrm{d} x^m}
                   
    .. math::
        S_n^m(x) = \sqrt{(2-\delta_m^0)\frac{(n-m)!}{(n+m)!}}P_n^m(x)

    Associated Legendre polynomials are orthogonal on [-1, 1] with:

    - weight = 1            for the same m, and different n.
    - weight = 1/(1-x**2)   for the same n, and different m.

    Examples
    ========

    >>> from assoc_legendre_schmidtnorm import assoc_legendre_schmidt_norm
    >>> from sympy.abc import x, theta, m, n
    >>> from sympy import cos
    >>> assoc_legendre_schmidt_norm(0,0, x)
    1
    >>> assoc_legendre_schmidt_norm(1,0, x)
    x
    >>> assoc_legendre_schmidt_norm(2,0, cos(theta))
    3*cos(theta)**2/2 - 1/2
    >>> assoc_legendre_schmidt_norm(4,0, cos(theta))
    35*cos(theta)**4/8 - 15*cos(theta)**2/4 + 3/8
    >>> assoc_legendre_schmidt_norm(n,m,x)
    assoc_legendre_schmidt_norm(n, m, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre,
    hermite,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    sympy.functions.special.polynomials.assoc_legendre

    References
    ==========

    .. [1] http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    .. [2] http://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] http://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] http://functions.wolfram.com/Polynomials/LegendreP2/
    .. [5] https://www.spenvis.oma.be/help/background/magfield/legendre.html#Schmidt1
    .. [6] Connerney Magnetc Fields of the Outer Planets (1993), J. Geophys. Res., Table 2 on p.661
    """
    
    print("Have imported assoc_legendre_schmidtnorm.py")

    @classmethod
    def _eval_at_order(cls, n, m):
        P = legendre_poly(n, _x, polys=True).diff((_x, m))
        return sqrt(2*factorial(n - m)/factorial(n + m)) *(1 - _x**2)**Rational(m, 2) * P.as_expr() # NOTE : no Condon-Shortley phase!
    @classmethod
    def eval(cls, n, m, x):
        if m.could_extract_minus_sign():
            # print("negative m")
            # P^{-m}_n  --->  F * P^m_n
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre_schmidt_norm(n, -m, x) # defn for P^{-m}_l(x) with m-> -m
        if m == 0:
            # for m=0, associated legendre polynomial is identical to legendre polynomial
            # P^0_n  --->L_n
            return legendre(n, x)
        if x == 0:
            # Unsure where this formula comes from, so I cannot define a Schmidt-quasi-normalised formula for x=0, 
            # however I shouldn't need to set x=0 as I want the expression as a function of cos(theta) for any non-zero theta 
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x) # evaluate the associated legendre expression, subbing in x for _x above

    def fdiff(self, argindex=3):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt m
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = 1
            n, m, x = self.args
            return 1/(x**2 - 1)*(x*n*assoc_legendre_schmidt_norm(n, m, x) - (m + n)*assoc_legendre_schmidt_norm(n - 1, m, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_polynomial(self, n, m, x):
        from sympy import Sum
        k = Dummy("k")
        kern = factorial(2*n - 2*k)/(2**n*factorial(n - k)*factorial(
            k)*factorial(n - 2*k - m))*(-1)**k*x**(n - m - 2*k)
        return (1 - x**2)**(m/2) * Sum(kern, (k, 0, floor((n - m)*S.Half)))

    def _eval_conjugate(self):
        n, m, x = self.args
        return self.func(n, m.conjugate(), x.conjugate())
    
