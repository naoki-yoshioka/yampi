#ifndef YAMPI_TEST_SOME
# define YAMPI_TEST_SOME

# include <boost/config.hpp>

# include <vector>
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
  template <typename ContiguousIterator>
  inline bool test_some(
    ContiguousIterator const first, ContiguousIterator const last,
    std::vector<int>& indices, std::vector< ::yampi::status >& statuses,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::request>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::request");

    int const size = last-first;
    indices.resize(size);
    statuses.resize(size);

    int num_completed_requests;
    int const error_code
      = MPI_Waitsome(
          size, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)),
          YAMPI_addressof(num_completed_requests),
          YAMPI_addressof(indices.front()),
          reinterpret_cast<MPI_Status*>(YAMPI_addressof(statuses.front())));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::test_some", environment);

    if (num_completed_requests == MPI_UNDEFINED)
      return false;

    indices.resize(num_completed_requests);
    statuses.resize(num_completed_requests);
    return true;
  }

  template <typename ContiguousRange>
  inline bool test_some(
    ContiguousRange const& requests,
    std::vector<int>& indices, std::vector< ::yampi::status >& statuses,
    ::yampi::environment const& environment)
  { return ::yampi::test_some(boost::begin(requests), boost::end(requests), indices, statuses, environment); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_remove_volatile
# undef YAMPI_is_same

#endif

