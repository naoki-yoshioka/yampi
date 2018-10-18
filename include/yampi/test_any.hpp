#ifndef YAMPI_TEST_ANY
# define YAMPI_TEST_ANY

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
  template <typename ContiguousIterator>
  inline boost::optional< std::pair< ::yampi::status, ContiguousIterator > >
  test_any(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::request>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::request");

    MPI_Status mpi_status;
    int index;
    int flag;
    int const error_code
      = MPI_Testany(
          last-first, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)),
          YAMPI_addressof(index), YAMPI_addressof(flag), YAMPI_addressof(mpi_status));

    typedef std::pair< ::yampi::status, ContiguousIterator > result_type;
    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? index != MPI_UNDEFINED
          ? boost::make_optional(result_type(::yampi::stauts(mpi_status), first + index))
          : boost::make_optional(result_type(::yampi::stauts(mpi_status), last))
        : boost::none
      : throw ::yampi::error(error_code, "yampi::test_any", environment);
  }

  template <typename ContiguousRange>
  inline
  boost::optional<
    std::pair<
      ::yampi::status,
      typename boost::range_iterator<ContiguousRange const>::type > >
  test_any(
    ContiguousRange const& requests, ::yampi::environment const& environment)
  { return ::yampi::test_any(boost::begin(requests), boost::end(requests), environment); }

  template <typename ContiguousIterator>
  inline boost::optional<ContiguousIterator> test_any(
    ::yampi::ignore_status_t const,
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::request>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::request");

    int index;
    int flag;
    int const error_code
      = MPI_Testany(
          last-first, reinterpret_cast<MPI_Request*>(YAMPI_addressof(*first)),
          YAMPI_addressof(index), YAMPI_addressof(flag), MPI_STATUS_IGNORE);

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
        ? index != MPI_UNDEFINED
          ? boost::make_optional(first + index)
          : boost::make_optional(last)
        : boost::none
      : throw ::yampi::error(error_code, "yampi::test_any", environment);
  }

  template <typename ContiguousRange>
  inline
  boost::optional<typename boost::range_iterator<ContiguousRange const>::type>
  test_any(
    ::yampi::ignore_status_t const ignore_status,
    ContiguousRange const& requests, ::yampi::environment const& environment)
  { return ::yampi::test_any(ignore_status, boost::begin(requests), boost::end(requests), environment); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_remove_volatile
# undef YAMPI_is_same

#endif

