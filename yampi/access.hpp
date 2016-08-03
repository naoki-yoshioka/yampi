#ifndef YAMPI_ACCESS_HPP
# define YAMPI_ACCESS_HPP

# include <yampi/datatype.hpp>


namespace yampi
{
  struct access
  {
    template <typename Value>
    static void derive_datatype(Value const& value)
    { value.derive_datatype(); }
  };
}


#endif

