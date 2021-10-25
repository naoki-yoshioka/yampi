#ifndef YAMPI_COMPARE_SWAP_HPP
# define YAMPI_COMPARE_SWAP_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename Value, typename Integer, typename Derived>
  inline
  typename YAMPI_enable_if< ::yampi::has_predefined_datatype<Value>::value, void >::type
  compare_swap(
    Value const& origin_value, Value const& compare_value, Value& result_value,
    ::yampi::rank const& target, Integer const target_displacement,
    ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Compare_and_swap(
          YAMPI_addressof(origin_value), YAMPI_addressof(compare_value), YAMPI_addressof(result_value), ::yampi::predefined_datatype<Value>(),
          target.mpi_rank(), static_cast<MPI_Aint>(target_displacement),
          window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::compare_swap", environment);
  }
}
# endif // MPI_VERSION >= 3


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif
