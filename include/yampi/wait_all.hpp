#ifndef YAMPI_WAIT_ALL
# define YAMPI_WAIT_ALL

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

# include <boost/range/iterator.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/status.hpp>
# include <yampi/request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_remove_volatile boost::remove_volatile
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
  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline void wait_all(
    ContiguousIterator1 const first, ContiguousIterator1 const last,
    ContiguousIterator2 const out, ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
         ::yampi::request>::value),
      "Value type of ContiguousIterator1 must be the same to ::yampi::request");
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
         ::yampi::status>::value),
      "Value type of ContiguousIterator2 must be the same to ::yampi::status");

    int const error_code
      = MPI_Waitall(
          last-first, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)),
          reinterpret_cast<MPI_Status*>(YAMPI_addressof(*out)));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait_all", environment);
  }

  template <typename ContiguousRange, typename ContiguousIterator>
  inline void wait_all(
    ContiguousRange const& requests, ContiguousIterator const out,
    ::yampi::environment const& environment)
  { ::yampi::wait_all(boost::begin(requests), boost::end(requests), out, environment); }

  template <typename ContiguousIterator>
  inline void wait_all(
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
      = MPI_Waitall(
          last-first, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)),
          MPI_STATUSES_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait_all", environment);
  }

  template <typename ContiguousRange>
  inline void wait_all(
    ContiguousRange const& requests, ::yampi::environment const& environment)
  { ::yampi::wait_all(boost::begin(requests), boost::end(requests), environment); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_remove_volatile
# undef YAMPI_is_same

#endif

