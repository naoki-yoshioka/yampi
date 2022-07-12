#ifndef YAMPI_ROOT_CALL_ON_NONROOT_ERROR_HPP
# define YAMPI_ROOT_CALL_ON_NONROOT_ERROR_HPP

# include <stdexcept>
# include <string>


namespace yampi
{
  class root_call_on_nonroot_error
    : public std::logic_error
  {
   public:
    root_call_on_nonroot_error(std::string const& where)
      : std::logic_error{(std::string("Root call on non-root in ") + where).c_str()}
    { }
  };
}


#endif

