#ifndef YAMPI_GROUP_HPP
# define YAMPI_GROUP_HPP

# include <iterator>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/rank.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct empty_group_t { };

  struct make_union_t { };
  struct make_intersection_t { };
  struct make_difference_t { };
  struct make_inclusive_t { };
  struct make_exclusive_t { };

  class strided_rank_range
  {
    int data_[3];

   public:
    constexpr strided_rank_range(::yampi::rank const first_rank, ::yampi::rank const last_rank, int const stride = 1) noexcept
      : data_()
    {
      data_[0] = first_rank.mpi_rank();
      data_[1] = last_rank.mpi_rank();
      data_[2] = stride;
    }

    constexpr ::yampi::rank first_rank() const noexcept { return ::yampi::rank(data_[0]); }
    constexpr ::yampi::rank last_rank() const noexcept { return ::yampi::rank(data_[1]); }
    constexpr int stride() const noexcept { return data_[2]; }

    void swap(strided_rank_range& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
    }
  }; // class strided_rank_range

  inline constexpr bool operator==(
    ::yampi::strided_rank_range const& lhs, ::yampi::strided_rank_range const& rhs)
    noexcept
  { return lhs.first_rank() == rhs.first_rank() and lhs.last_rank() == rhs.last_rank() and lhs.stride() == rhs.stride(); }

  inline constexpr bool operator!=(
    ::yampi::strided_rank_range const& lhs, ::yampi::strided_rank_range const& rhs)
    noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::strided_rank_range& lhs, ::yampi::strided_rank_range& rhs) noexcept
  { lhs.swap(rhs); }

  class group
  {
    MPI_Group mpi_group_;

   public:
    group() noexcept(std::is_nothrow_copy_constructible<MPI_Group>::value)
      : mpi_group_(MPI_GROUP_NULL)
    { }

    group(group const&) = delete;
    group& operator=(group const&) = delete;

    group(group&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Group>::value
        and std::is_nothrow_copy_assignable<MPI_Group>::value)
      : mpi_group_(std::move(other.mpi_group_))
    { other.mpi_group_ = MPI_GROUP_NULL; }

    group& operator=(group&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Group>::value
        and std::is_nothrow_copy_assignable<MPI_Group>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_group_ != MPI_GROUP_NULL and mpi_group_ != MPI_GROUP_EMPTY)
          MPI_Group_free(std::addressof(mpi_group_));
        mpi_group_ = std::move(other.mpi_group_);
        other.mpi_group_ = MPI_GROUP_NULL;
      }
      return *this;
    }

    ~group() noexcept
    {
      if (mpi_group_ == MPI_GROUP_NULL or mpi_group_ == MPI_GROUP_EMPTY)
        return;

      MPI_Group_free(std::addressof(mpi_group_));
    }

    explicit group(MPI_Group const& mpi_group)
      noexcept(std::is_nothrow_copy_constructible<MPI_Group>::value)
      : mpi_group_(mpi_group)
    { }

    explicit group(::yampi::empty_group_t const)
      noexcept(std::is_nothrow_copy_constructible<MPI_Group>::value)
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
      : mpi_group_(make_inclusive(original_group, std::begin(ranks), std::end(ranks), environment))
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
      : mpi_group_(make_exclusive(original_group, std::begin(ranks), std::end(ranks), environment))
    { }

   private:
    MPI_Group make_union(
      group const& lhs, group const& rhs,
      ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_union(lhs.mpi_group(), rhs.mpi_group(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::group::make_union", environment);
    }

    MPI_Group make_intersection(
      group const& lhs, group const& rhs,
      ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_intersection(
            lhs.mpi_group(), rhs.mpi_group(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_intersection", environment);
    }

    MPI_Group make_difference(
      group const& lhs, group const& rhs,
      ::yampi::environment const& environment) const
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_difference(
            lhs.mpi_group(), rhs.mpi_group(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_difference", environment);
    }

    template <typename ContiguousIterator>
    typename std::enable_if<
      std::is_same<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type,
        ::yampi::rank>::value,
      MPI_Group>::type
    make_inclusive(
      group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_incl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int const*>(std::addressof(*first)),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_inclusive", environment);
    }

    template <typename ContiguousIterator>
    typename std::enable_if<
      std::is_same<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type,
        ::yampi::strided_rank_range>::value,
      MPI_Group>::type
    make_inclusive(
      group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_range_incl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int [][3]>(std::addressof(*first)),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_inclusive", environment);
    }

    template <typename ContiguousIterator>
    typename std::enable_if<
      std::is_same<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type,
        ::yampi::rank>::value,
      MPI_Group>::type
    make_exclusive(
      group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_excl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int const*>(std::addressof(*first)),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_exclusive", environment);
    }

    template <typename ContiguousIterator>
    typename std::enable_if<
      std::is_same<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type,
        ::yampi::strided_rank_range>::value,
      MPI_Group>::type
    make_exclusive(
      group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      MPI_Group result;
      int const error_code
        = MPI_Group_range_excl(
            original_group.mpi_group(), last-first,
            reinterpret_cast<int [][3]>(std::addressof(*first)),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::group::make_exclusive", environment);
    }

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Group const& mpi_group, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = mpi_group;
    }

    void reset(group&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = std::move(other.mpi_group_);
      other.mpi_group_ = MPI_GROUP_NULL;
    }

    void reset(::yampi::empty_group_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = MPI_GROUP_EMPTY;
    }

    void reset(
      ::yampi::make_union_t const, group const& lhs, group const& rhs,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = make_union(lhs, rhs, environment);
    }

    void reset(
      ::yampi::make_intersection_t const, group const& lhs, group const& rhs,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = make_intersection(lhs, rhs, environment);
    }

    void reset(
      ::yampi::make_difference_t const, group const& lhs, group const& rhs,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = make_difference(lhs, rhs, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ::yampi::make_inclusive_t const, group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = make_inclusive(original_group, first, last, environment);
    }

    template <typename ContiguousRange>
    void reset(
      ::yampi::make_inclusive_t const,
      group const& original_group, ContiguousRange const& ranks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_
        = make_inclusive(
            original_group, std::begin(ranks), std::end(ranks), environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ::yampi::make_exclusive_t const, group const& original_group,
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_ = make_exclusive(original_group, first, last, environment);
    }

    template <typename ContiguousRange>
    void reset(
      ::yampi::make_exclusive_t const,
      group const& original_group, ContiguousRange const& ranks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_group_
        = make_exclusive(
            original_group, std::begin(ranks), std::end(ranks), environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_group_ == MPI_GROUP_NULL or mpi_group_ == MPI_GROUP_EMPTY)
        return;

      int const error_code = MPI_Group_free(&mpi_group_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::group::free", environment);
    }

    bool is_null() const noexcept(noexcept(mpi_group_ == MPI_GROUP_NULL))
    { return mpi_group_ == MPI_GROUP_NULL; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Group_size(mpi_group_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::group::size", environment);
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Group_rank(mpi_group_, std::addressof(mpi_rank));
      return error_code == MPI_SUCCESS
        ? ::yampi::rank(mpi_rank)
        : throw ::yampi::error(error_code, "yampi::group::rank", environment);
    }

    MPI_Group const& mpi_group() const noexcept { return mpi_group_; }

    void swap(group& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Group>::value)
    {
      using std::swap;
      swap(mpi_group_, other.mpi_group_);
    }
  };

  inline void swap(::yampi::group& lhs, ::yampi::group& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  template <typename ContiguousIterator1, typename ContiguousIterator2>
  inline void translate(
    ::yampi::group const& old_group, ContiguousIterator1 const first, ContiguousIterator1 const last,
    ::yampi::group const& new_group, ContiguousIterator2 out, ::yampi::environment const& environment)
  {
    static_assert(
      (std::is_same<
         typename std::remove_cv<
           typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
         ::yampi::rank>::value),
      "Value type of ContiguousIterator1 must be the same to ::yampi::rank");
    static_assert(
      (std::is_same<
         typename std::remove_volatile<
           typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
         ::yampi::rank>::value),
      "Value type of ContiguousIterator2 must be the same to ::yampi::rank");

    int const error_code
      = MPI_Group_translate_ranks(
          old_group.mpi_group(), last-first,
          reinterpret_cast<int const*>(std::addressof(*first)),
          new_group.mpi_group(), reinterpret_cast<int*>(std::addressof(*out)));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::translate", environment);
  }

  template <typename ContiguousRange, typename ContiguousIterator>
  inline void translate(
    ::yampi::group const& old_group, ContiguousRange const& old_ranks,
    ::yampi::group const& new_group, ContiguousIterator out,
    ::yampi::environment const& environment)
  { translate(old_group, std::begin(old_ranks), std::end(old_ranks), new_group, out, environment); }
}


# undef YAMPI_is_nothrow_swappable

#endif

