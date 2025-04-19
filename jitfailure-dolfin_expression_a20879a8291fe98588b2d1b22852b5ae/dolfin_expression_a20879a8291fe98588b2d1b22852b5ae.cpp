
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_a20879a8291fe98588b2d1b22852b5ae : public Expression
  {
     public:
       double mu_air;
double mu_electromagnet;
double mu_metal_sheet;
double height;


       dolfin_expression_a20879a8291fe98588b2d1b22852b5ae()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = x[2] <= 0.5 * height ? (subdomain == 1 ? mu_electromagnet : (subdomain == 2 ? mu_metal_sheet : mu_air)) : mu_air;

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "mu_air") { mu_air = _value; return; }          if (name == "mu_electromagnet") { mu_electromagnet = _value; return; }          if (name == "mu_metal_sheet") { mu_metal_sheet = _value; return; }          if (name == "height") { height = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "mu_air") return mu_air;          if (name == "mu_electromagnet") return mu_electromagnet;          if (name == "mu_metal_sheet") return mu_metal_sheet;          if (name == "height") return height;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_a20879a8291fe98588b2d1b22852b5ae()
{
  return new dolfin::dolfin_expression_a20879a8291fe98588b2d1b22852b5ae;
}

