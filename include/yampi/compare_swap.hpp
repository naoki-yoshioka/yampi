#ifndef YAMPI_COMPARE_SWAP_HPP
# define YAMPI_COMPARE_SWAP_HPP

# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/displacement.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename Value, typename Derived>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<Value>::value, void >::type
  compare_swap(
    Value const& origin_value, Value const& compare_value, Value& result_value,
    ::yampi::rank const target, ::yampi::displacement const target_displacement,
    ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Compare_and_swap(
          std::addressof(origin_value), std::addressof(compare_value), std::addressof(result_value), ::yampi::predefined_datatype<Value>(),
          target.mpi_rank(), target_displacement.mpi_displacement(),
          window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::compare_swap", environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

