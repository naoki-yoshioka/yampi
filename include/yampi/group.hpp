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

  struct make_union_t { };
  struct make_intersection_t { };
  struct make_difference_t { };
  struct make_inclusive_t { };
  struct make_exclusive_t { };

  class group
  {
    MPI_Group mpi_group_;

   public:
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    group() = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    group() : mpi_group_() { }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    group(group const&) = delete;
    group& operator=(group const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    group(group const&);
    group& operator=(group const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    group(group&&) = default;
    group& operator=(group&&) = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    group(group&& other) : mpi_group_(std::move(other.mpi_group_)) { }
    group& operator=(group&& other)
    {
      if (this != YAMPI_addressof(other))
        mpi_group_ = std::move(other.mpi_group_);
      return *this;
    }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~group() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_group_ == MPI_GROUP_NULL or mpi_group_ == MPI_GROUP_EMPTY or mpi_group_ == MPI_Group())
        return;

      MPI_Group_free(YAMPI_addressof(mpi_group_));
    }

    explicit group(MPI_Group const mpi_group)
      : mpi_group_(mpi_group)
    { }

    explicit group(::yampi::empty_group_t const)
      : mpi_group_(MPI_GROUP_EMPTY)
    { }

    group(::yampi::make_union_t const, group const& lhs, group const& rhs, ::yampi::environment const& environment)
      : mpi_group_(make_union(lhs, rhs, environment))
    { }

    group(::yampi::make_intersection_t const, group const& lhs, group const& rhs, ::yampi::environment const& environment)
      : mpi_group_(make_intersection(lhs, rhs, environment))
    { }

    group(::yampi::make_difference_t const, group const& lhs, group const& rhs, ::yampi::environment const& environment)
      : mpi_group_(make_difference(lhs, rhs, environment))
    { }

    // TODO: Implement range versions
    template <typename ContiguousIterator>
    group(
      ::yampi::make_inclusive_t const,
      group const& original_group, ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
      : mpi_group_(make_inclusive(original_group, first, last, environment))
    { }

    template <typename ContiguousRange>
    group(
      ::yampi::make_inclusive_t const,
      group const& original_group, ContiguousRange const& ranks,
      ::yampi::environment const& environment)
      : mpi_group_(make_inclusive(original_group, boost::begin(ranks), boost::end(ranks), environment))
    { }

    template <typename ContiguousIterator>
    group(
      ::yampi::make_exclusive_t const,
      group const& original_group, ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
      : mpi_group_(make_exclusive(original_group, first, last, environment))
    { }

    template <typename ContiguousRange>
    group(
      ::yampi::make_exclusive_t const,
      group const& original_group, ContiguousRange const& ranks,
      ::yampi::environment const& environment)
      : mpi_group_(make_exclusive(original_group, boost::begin(ranks), boost::end(ranks), environment))
    { }

   private:
    MPI_Group make_union(group const& lhs, group const& rhs, ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_union(lhs.mpi_group(), rhs.mpi_group(), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::make_union", environment);
      return result;
    }

    MPI_Group make_intersection(group const& lhs, group const& rhs, ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_intersection(lhs.mpi_group(), rhs.mpi_group(), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::make_intersection", environment);
      return result;
    }

    MPI_Group make_difference(group const& lhs, group const& rhs, ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_difference(lhs.mpi_group(), rhs.mpi_group(), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::make_difference", environment);
      return result;
    }

    template <typename ContiguousIterator>
    MPI_Group make_inclusive(
      group const& original_group, ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           ::yampi::rank>::value),
        "Value type of ContiguousIterator must be the same to ::yampi::rank");

      MPI_Group result;
      int const error_code
        = MPI_Group_incl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int const*>(YAMPI_addressof(*first)),
            YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::make_inclusive", environment);
      return result;
    }

    template <typename ContiguousIterator>
    MPI_Group make_exclusive(
      group const& original_group, ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           ::yampi::rank>::value),
        "Value type of ContiguousIterator must be the same to ::yampi::rank");

      MPI_Group result;
      int const error_code
        = MPI_Group_excl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int const*>(YAMPI_addressof(*first)),
            YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::make_exclusive", environment);
      return result;
    }

   public:
    void release(::yampi::environment const& environment)
    {
      if (mpi_group_ == MPI_GROUP_NULL or mpi_group_ == MPI_GROUP_EMPTY or mpi_group_ == MPI_Group())
        return;

      int const error_code = MPI_Group_free(&mpi_group_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::release", environment);
    }


    bool is_null() const { return mpi_group_ == MPI_GROUP_NULL; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Group_size(mpi_group_, YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::size", environment);
      return result;
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Group_rank(mpi_group_, YAMPI_addressof(mpi_rank));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::rank", environment);
      return ::yampi::rank(mpi_rank);
    }

    MPI_Group const& mpi_group() const { return mpi_group_; }
    void mpi_group(MPI_Group const& mpi_grp) { mpi_group_ = mpi_grp; }

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


  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline void translate_ranks(
    ::yampi::group const& old_group, ContiguousIterator1 const first, ContiguousIterator1 const last,
    ::yampi::group const& new_group, ContiguousIterator2 out, ::yampi::environment const& environment)
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
          old_group.mpi_group(), last-first,
          reinterpret_cast<int const*>(YAMPI_addressof(*first)),
          new_group.mpi_group(), reinterpret_cast<int*>(YAMPI_addressof(*out)));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::translate_ranks", environment);
  }

  template <typename ContiguousRange, typename ContiguousIterator>
  inline void translate_ranks(
    ::yampi::group const& old_group, ContiguousRange const& old_ranks,
    ::yampi::group const& new_group, ContiguousIterator out,
    ::yampi::environment const& environment)
  { translate(old_group, boost::begin(old_ranks), boost::end(old_ranks), new_group, out, environment); }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_remove_cv
# undef YAMPI_remove_volatile
# undef YAMPI_is_same


#endif

