#ifndef YAMPI_RANK_HPP
# define YAMPI_RANK_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct host_process_t { };
  struct io_process_t { };
  struct any_source_t { };
  struct null_process_t { };

  namespace tags
  {
# if __cplusplus >= 201703L
    inline constexpr ::yampi::host_process_t host_process{};
    inline constexpr ::yampi::io_process_t io_process{};
    inline constexpr ::yampi::any_source_t any_source{};
    inline constexpr ::yampi::null_process_t null_process{};
# else
    constexpr ::yampi::host_process_t host_process{};
    constexpr ::yampi::io_process_t io_process{};
    constexpr ::yampi::any_source_t any_source{};
    constexpr ::yampi::null_process_t null_process{};
# endif
  }

  class rank
  {
    int mpi_rank_;

   public:
    constexpr rank() noexcept : mpi_rank_{0} { }

    explicit constexpr rank(int const mpi_rank) noexcept : mpi_rank_{mpi_rank} { }

    explicit constexpr rank(::yampi::any_source_t const) noexcept
      : mpi_rank_{MPI_ANY_SOURCE}
    { }

    explicit constexpr rank(::yampi::null_process_t const) noexcept
      : mpi_rank_{MPI_PROC_NULL}
    { }

    explicit rank(::yampi::host_process_t const, ::yampi::environment const& environment)
      : mpi_rank_{inquire(MPI_HOST, environment)}
    { }

    explicit rank(::yampi::io_process_t const, ::yampi::environment const& environment)
      : mpi_rank_{inquire(MPI_IO, environment)}
    { }

   private:
    int inquire(int const key_value, ::yampi::environment const& environment) const
    {
      // don't check flag because users cannnot delete the attribute MPI_HOST
      int* result;
      int flag;
      int const error_code = MPI_Comm_get_attr(MPI_COMM_WORLD, key_value, &result, &flag);

      return error_code == MPI_SUCCESS
        ? *result
        : throw ::yampi::error(error_code, "yampi::rank::inquire_environment", environment);
    }

   public:
    rank(rank const&) = default;
    rank& operator=(rank const&) = default;
    rank(rank&&) = default;
    rank& operator=(rank&&) = default;
    ~rank() noexcept = default;

    bool is_null() const noexcept { return mpi_rank_ == MPI_PROC_NULL; }

    constexpr bool operator==(rank const& other) const noexcept
    { return mpi_rank_ == other.mpi_rank_; }

    bool operator<(rank const& other) const noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(
        other.mpi_rank_ != MPI_ANY_SOURCE and other.mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return mpi_rank_ < other.mpi_rank_;
    }

    rank& operator++() noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      ++mpi_rank_;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL);
      return *this;
    }

    rank& operator--() noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      --mpi_rank_;
      assert(mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      rank&>::type
    operator+=(Integer const n) noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      mpi_rank_ += n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      rank&>::type
    operator-=(Integer const n) noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      mpi_rank_ -= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      rank&>::type
    operator*=(Integer const n) noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_rank_ *= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      rank&>::type
    operator/=(Integer const n) noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_rank_ /= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      rank&>::type
    operator%=(Integer const n) noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_rank_ %= n;
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return *this;
    }

    int operator-(rank const& other) const noexcept
    {
      assert(mpi_rank_ != MPI_ANY_SOURCE and mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      assert(
        other.mpi_rank_ != MPI_ANY_SOURCE and other.mpi_rank_ != MPI_PROC_NULL and mpi_rank_ >= 0);
      return mpi_rank_-other.mpi_rank_;
    }

    constexpr int const& mpi_rank() const noexcept { return mpi_rank_; }
    explicit constexpr operator int() const { return mpi_rank_; }

    void swap(rank& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_rank_, other.mpi_rank_);
    }
  };

  inline constexpr bool operator!=(::yampi::rank const& lhs, ::yampi::rank const& rhs) noexcept
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::rank const& lhs, ::yampi::rank const& rhs) noexcept
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::rank const& lhs, ::yampi::rank const& rhs) noexcept
  { return rhs < lhs; }

  inline bool operator<=(::yampi::rank const& lhs, ::yampi::rank const& rhs) noexcept
  { return not (rhs < lhs); }

  inline ::yampi::rank operator++(::yampi::rank& lhs, int) noexcept
  { ::yampi::rank result = lhs; ++lhs; return result; }

  inline ::yampi::rank operator--(::yampi::rank& lhs, int) noexcept
  { ::yampi::rank result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::rank operator+(::yampi::rank lhs, Integer const rhs) noexcept
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator-(::yampi::rank lhs, Integer const rhs) noexcept
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator*(::yampi::rank lhs, Integer const rhs) noexcept
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator/(::yampi::rank lhs, Integer const rhs) noexcept
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator%(::yampi::rank lhs, Integer const rhs) noexcept
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::rank operator+(Integer const lhs, ::yampi::rank const& rhs) noexcept
  { return rhs+lhs; }

  template <typename Integer>
  inline ::yampi::rank operator*(Integer const lhs, ::yampi::rank const& rhs) noexcept
  { return rhs*lhs; }

  inline void swap(::yampi::rank& lhs, ::yampi::rank& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }


# if __cplusplus >= 201703L
  inline constexpr ::yampi::rank any_source{::yampi::tags::any_source};
  inline constexpr ::yampi::rank null_process{::yampi::tags::null_process};
# else
  constexpr ::yampi::rank any_source{::yampi::tags::any_source};
  constexpr ::yampi::rank null_process{::yampi::tags::null_process};
# endif


  inline ::yampi::rank host_process(::yampi::environment const& environment)
  { return ::yampi::rank(::yampi::host_process_t(), environment); }

  inline ::yampi::rank io_process(::yampi::environment const& environment)
  { return ::yampi::rank(::yampi::io_process_t(), environment); }


  inline bool exists_host_process(::yampi::environment const& environment)
  { return not ::yampi::host_process(environment).is_null(); }

  inline bool is_host_process(::yampi::rank const& self, ::yampi::environment const& environment)
  { return self == ::yampi::host_process(environment); }

  inline bool exists_io_process(::yampi::environment const& environment)
  { return not ::yampi::io_process(environment).is_null(); }

  inline bool is_io_process(::yampi::rank const& self, ::yampi::environment const& environment)
  {
    ::yampi::rank const io = ::yampi::io_process(environment);
    return io == ::yampi::any_source or self == io;
  }
}


# undef YAMPI_is_nothrow_swappable

#endif

