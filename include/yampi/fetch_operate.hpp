#ifndef YAMPI_FETCH_OPERATE_HPP
# define YAMPI_FETCH_OPERATE_HPP

# include <cassert>
# include <memory>
# include <type_traits>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/displacement.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename Value, typename Derived>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<Value>::value, void >::type
  fetch_operate(
    Value const& origin_value, Value& result_value,
    ::yampi::rank const target, ::yampi::displacement const target_displacement,
    ::yampi::binary_operation const& operation,
    ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    assert(std::addressof(origin_value) != std::addressof(result_value));

    int const error_code
      = MPI_Fetch_and_op(
          std::addressof(origin_value), std::addressof(result_value), ::yampi::predefined_datatype<Value>(),
          target.mpi_rank(), target_displacement.mpi_displacement(),
          operation.mpi_op(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::fetch_operate", environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

