#ifndef YAMPI_RANK_HPP
# define YAMPI_RANK_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  class any_source_t { };
  class null_process_t { };

  class rank
  {
    int mpi_rank_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    BOOST_CONSTEXPR rank() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_{0}
    { }

    explicit BOOST_CONSTEXPR rank(int const mpi_rank) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_{mpi_rank}
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::any_source_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_{MPI_ANY_SOURCE}
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::null_process_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_{MPI_PROC_NULL}
    { }
# else
    BOOST_CONSTEXPR rank() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(0)
    { }

    explicit BOOST_CONSTEXPR rank(int const mpi_rank) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(mpi_rank)
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::any_source_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(MPI_ANY_SOURCE)
    { }

    explicit BOOST_CONSTEXPR rank(::yampi::null_process_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_rank_(MPI_PROC_NULL)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    rank(rank const&) = default;
    rank& operator=(rank const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    rank(rank&&) = default;
    rank& operator=(rank&&) = default;
#   endif
    ~rank() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    rank& operator++() { ++mpi_rank_; return *this; }
    rank& operator++(int) { mpi_rank_++; return *this; }
    rank& operator--() { --mpi_rank_; return *this; }
    rank& operator--(int) { mpi_rank_--; return *this; }
    rank& operator+=(rank const other) { mpi_rank_ += other.mpi_rank_; return *this; }
    rank& operator-=(rank const other) { mpi_rank_ -= other.mpi_rank_; return *this; }
    rank& operator*=(rank const other) { mpi_rank_ *= other.mpi_rank_; return *this; }
    rank& operator/=(rank const other) { mpi_rank_ /= other.mpi_rank_; return *this; }

    bool operator==(rank const ohter) const { return mpi_rank_ == other.mpi_rank_; }
    bool operator<(rank const ohter) const { return mpi_rank_ < other.mpi_rank_; }

    int mpi_rank() const { return mpi_rank_; }
  };

  inline bool operator!=(::yampi::rank const lhs, ::yampi::rank const rhs)
  { return !(lhs == rhs); }

  inline bool operator>=(::yampi::rank const lhs, ::yampi::rank const rhs)
  { return !(lhs < rhs); }

  inline bool operator>(::yampi::rank const lhs, ::yampi::rank const rhs)
  { return rhs < lhs; }

  inline bool operator<=(::yampi::rank const lhs, ::yampi::rank const rhs)
  { return !(rhs < lhs); }

  inline ::yampi::rank operator+(::yampi::rank lhs, ::yampi::rank const rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::rank operator-(::yampi::rank lhs, ::yampi::rank const rhs)
  { lhs -= rhs; return lhs; }

  inline ::yampi::rank operator*(::yampi::rank lhs, ::yampi::rank const rhs)
  { lhs *= rhs; return lhs; }

  inline ::yampi::rank operator/(::yampi::rank lhs, ::yampi::rank const rhs)
  { lhs /= rhs; return lhs; }

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  auto const any_source = ::yampi::rank{::yampi::any_source_t{}};
  auto const null_process = ::yampi::rank{::yampi::null_process_t{}};
#   else
  auto const any_source = ::yampi::rank(::yampi::any_source_t());
  auto const null_process = ::yampi::rank(::yampi::null_process_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  ::yampi::rank const any_source{::yampi::any_source_t{}};
  ::yampi::rank const null_process{::yampi::null_process_t{}};
#   else
  ::yampi::rank const any_source(::yampi::any_source_t());
  ::yampi::rank const null_process(::yampi::null_process_t());
#   endif
# endif
}


#endif

