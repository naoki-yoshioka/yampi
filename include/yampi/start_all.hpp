#ifndef YAMPI_START_ALL
# define YAMPI_START_ALL

# include <boost/config.hpp>

# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  template <typename ContiguousIterator>
  inline void start_all(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::request>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::request");

    int const error_code
      = MPI_Startall(last-first, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::start_all", environment);
  }

  template <typename ContiguousRange>
  inline void start_all(
    ContiguousRange const& requests, ::yampi::environment const& environment)
  { ::yampi::start_all(boost::begin(requests), boost::end(requests), environment); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_is_same

#endif

