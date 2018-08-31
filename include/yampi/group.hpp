#ifndef YAMPI_GROUP_HPP
# define YAMPI_GROUP_HPP

# include <boost/config.hpp>

# include <iterator>
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

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_remove_volatile std::remove_volatile
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
  struct empty_group_t { };

  class group
  {
    MPI_Group mpi_group_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    group() = delete;
# else
   private:
    group();

   public:
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    group(group const&) = default;
    group& operator=(group const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    group(group&&) = default;
    group& operator=(group&&) = default;
#   endif
# endif

    ~group() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_group_ == MPI_GROUP_NULL)
        return;

      MPI_Group_free(&mpi_group_);
    }

    explicit group(MPI_Group const mpi_group)
      : mpi_group_(mpi_group)
    { }

    explicit group(::yampi::empty_group_t const)
      : mpi_group_(MPI_GROUP_EMPTY)
    { }

    group(::yampi::communicator const communicator, ::yampi::environment const& environment)
      : mpi_group_(generate_mpi_group(communicator, environment))
    { }

   private:
    MPI_Group generate_mpi_group(::yampi::communicator const communicator, ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code = MPI_Comm_group(communicator.mpi_comm(), &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::generate_mpi_group", environment);
      return result;
    }

   public:
    void release(::yampi::environment const& environment)
    {
      int const error_code = MPI_Group_free(&mpi_group_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::size", environment);
      return result;
    }


    bool is_null() const { return mpi_group_ == MPI_GROUP_NULL; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Group_size(mpi_group_, &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::size", environment);
      return result;
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Group_rank(mpi_group_, &mpi_rank);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::rank", environment);
      return ::yampi::rank(mpi_rank);
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    void translate_ranks(
      group const& original_group, ContiguousIterator1 const first, ContiguousIterator1 const last,
      ContiguousIterator2 out, ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
           ::yampi::rank>::value),
        "Value type of ContiguousIterator1 must be the same to ::yampi::rank");
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_volatile<
             typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
           ::yampi::rank>::value),
        "Value type of ContiguousIterator2 must be the same to ::yampi::rank");

      int const error_code
        = MPI_Group_translate_ranks(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int const*>(YAMPI_addressof(*first)),
            mpi_group_, reinterpret_cast<int*>(YAMPI_addressof(*out)));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::translate", environment);
    }

    template <typename ContiguousRange, typename ContiguousIterator>
    void translate_ranks(
      group const& original_group, ContiguousRange const& known_ranks, ContiguousIterator out,
      ::yampi::environment const& environment) const
    { translate(original_group, boost::begin(known_ranks), boost::end(known_ranks), out, environment); }

    MPI_Group const& mpi_group() const { return mpi_group_; }

    void swap(group& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Group>::value ))
    {
      using std::swap;
      swap(mpi_group_, other.mpi_group_);
    }
  };

  inline void swap(::yampi::group& lhs, ::yampi::group& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::group >::value ))
  { lhs.swap(rhs); }


  inline ::yampi::group empty_group()
  { return ::yampi::group(::yampi::empty_group_t()); }


  inline ::yampi::group make_union(
    ::yampi::group const lhs, ::yampi::group const rhs, ::yampi::environment const& environment)
  {
    MPI_Group mpi_group;
    int const error_code = MPI_Group_union(lhs.mpi_group(), rhs.mpi_group(), &mpi_group);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::make_union", environment);
    return ::yampi::group(mpi_group);
  }

  inline ::yampi::group make_intersection(
    ::yampi::group const lhs, ::yampi::group const rhs, ::yampi::environment const& environment)
  {
    MPI_Group mpi_group;
    int const error_code = MPI_Group_intersection(lhs.mpi_group(), rhs.mpi_group(), &mpi_group);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::make_intersection", environment);
    return ::yampi::group(mpi_group);
  }

  inline ::yampi::group make_difference(
    ::yampi::group const lhs, ::yampi::group const rhs, ::yampi::environment const& environment)
  {
    MPI_Group mpi_group;
    int const error_code = MPI_Group_difference(lhs.mpi_group(), rhs.mpi_group(), &mpi_group);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::make_difference", environment);
    return ::yampi::group(mpi_group);
  }

  // TODO: Implement range versions
  template <typename ContiguousIterator>
  inline ::yampi::group make_inclusive(
    ::yampi::group const original, ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::rank>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::rank");

    MPI_Group mpi_group;
    int const error_code
      = MPI_Group_incl(
          original_group.mpi_group(), last-first,
          reinterpret_cast<int const*>(YAMPI_addressof(*first)),
          mpi_group);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::group::translate", environment);
    return ::yampi::group(mpi_group);
  }

  template <typename ContiguousRange>
  inline ::yampi::group make_inclusive(
    ::yampi::group const original, ContiguousRange const& ranks, ::yampi::environment const& environment)
  { return ::yampi::make_inclusive(original, boost::begin(ranks), boost::end(ranks), environment); }

  template <typename ContiguousIterator>
  inline ::yampi::group make_exclusive(
    ::yampi::group const original, ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename YAMPI_remove_cv<
           typename std::iterator_traits<ContiguousIterator>::value_type>::type,
         ::yampi::rank>::value),
      "Value type of ContiguousIterator must be the same to ::yampi::rank");

    MPI_Group mpi_group;
    int const error_code
      = MPI_Group_excl(
          original_group.mpi_group(), last-first,
          reinterpret_cast<int const*>(YAMPI_addressof(*first)),
          mpi_group);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::group::translate", environment);
    return ::yampi::group(mpi_group);
  }

  template <typename ContiguousRange>
  inline ::yampi::group make_exclusive(
    ::yampi::group const original, ContiguousRange const& ranks, ::yampi::environment const& environment)
  { return ::yampi::make_exclusive(original, boost::begin(ranks), boost::end(ranks), environment); }
}


#endif

