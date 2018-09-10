#ifndef YAMPI_CARTESIAN_HPP
# define YAMPI_CARTESIAN_HPP

# include <boost/config.hpp>

# include <vector>
# include <iterator>
# include <algorithm>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/remove_volatile.hpp>
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

# include <yampi/topology.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/rank.hpp>

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
  class cartesian
    : public ::yampi::topology
  {
    using ::yampi::topology::communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    cartesian() = delete;
    cartesian(cartesian const&) = delete;
    cartesian& operator=(cartesian const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    cartesian();
    cartesian(cartesian const&);
    cartesian& operator=(cartesian const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    cartesian(cartesian&&) = default;
    cartesian& operator=(cartesian&&) = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    cartesian(cartesian&& other) : ::yampi::topology(std::move(other)) { }
    cartesian& operator=(cartesian&& other)
    {
      if (this != YAMPI_addressof(other))
        communicator_ = std::move(other.communicator_);
      return *this;
    }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~cartesian() = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~cartesian() { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    cartesian(
      ::yampi::environment const& environment,
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      bool const is_reorderable = true)
      : ::yampi::topology(
          create(
            environment, old_communicator, size_first, size_last,
            is_periodic_first, is_reorderable))
    { }

    template <typename ContiguousIterator>
    cartesian(
      cartesian const& other,
      ContiguousIterator const remains_first, ContiguousIterator const remains_last,
      ::yampi::environment const& environment)
      : ::yampi::topology(make_subcommunicator(other, remains_first, remains_last, environment))
    { }

   private:
    template <typename ContiguousIterator1, typename ContiguousIterator2>
    MPI_Comm create(
      ::yampi::environment const& environment,
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      bool const is_reorderable)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator2 must be the same to bool");

      std::vector<int> my_is_periodic(is_periodic_first, is_periodic_first + (size_last - size_first));

      MPI_Comm result;
      int const error_code
        = MPI_Cart_create(
            old_communicator.mpi_comm(),
            size_last-size_first, YAMPI_addressof(*size_first),
            YAMPI_addressof(my_is_periodic.front()),
            static_cast<int>(is_reorderable),
            YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::create", environment);
      return result;
    }

    template <typename ContiguousIterator>
    MPI_Comm make_subcommunicator(
      cartesian const& other,
      ContiguousIterator const remains_first, ContiguousIterator const remains_last,
      ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator must be the same to bool");

      std::vector<int> my_remains(remains_first, remains_last);

      MPI_Comm result;
      int const error_code
        = MPI_Cart_sub(
            other.communicator().mpi_comm(), YAMPI_addressof(my_remains.front()),
            YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::make_subcommunicator", environment);
      return result;
    }

   public:
    int dimension(::yampi::environment const& environment) const
    {
      int result;
      int const error_code
        = MPI_Cartdim_get(communicator_.mpi_comm(), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::dimension", environment);
      return result;
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
    void topology_information(
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      ContiguousIterator3 const coordinates_first,
      ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator2 must be the same to bool");
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator3>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator3 must be the same to int");

      std::vector<int> my_is_periodic(size_last - size_first);

      int const error_code
        = MPI_Cart_get(
            communicator_.mpi_comm(),
            size_last-size_first, YAMPI_addressof(*size_first),
            YAMPI_addressof(my_is_periodic.front()),
            YAMPI_addressof(*coordinates_first));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::topology_information", environment);
      std::transform(
        boost::begin(my_is_periodic), boost::end(my_is_periodic), is_periodic_first);
    }

    template <typename ContiguousIterator>
    ::yampi::rank rank(
      ContiguousIterator const coordinates_first, ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");

      int mpi_rank;
      int const error_code
        = MPI_Cart_rank(
            communicator_.mpi_comm(), YAMPI_addressof(*coordinates_first),
            YAMPI_addressof(mpi_rank));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::rank", environment);
      return ::yampi::rank(mpi_rank);
    }

    template <typename ContiguousIterator>
    void coordinates(
      ::yampi::rank const rank,
      ContiguousIterator const coordinates_first,
      ContiguousIterator const coordinates_last,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");

      int const error_code
        = MPI_Cart_coords(
            communicator_.mpi_comm(), rank.mpi_rank(),
            coordinates_last - coordinates_first,
            YAMPI_addressof(*coordinates_first));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::coordinates", environment);
    }

    void swap(cartesian& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::communicator >::value ))
    {
      using std::swap;
      swap(communicator_, other.communicator_);
    }
  };

  inline void swap(::yampi::cartesian& lhs, ::yampi::cartesian& rhs)
    BOOST_NOEXCEPT(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_is_same

#endif

