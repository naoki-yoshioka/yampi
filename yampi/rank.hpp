#ifndef YAMPI_RANK_HPP
# define YAMPI_RANK_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  struct host_process_t { };
  struct io_process_t { };
  struct any_source_t { };
  struct null_process_t { };

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

    explicit rank(::yampi::host_process_t const)
      : mpi_rank_{inquire_environment(MPI_HOST)}
    { }

    explicit rank(::yampi::io_process_t const)
      : mpi_rank_{inquire_environment(MPI_IO)}
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

    explicit rank(::yampi::host_process_t const)
      : mpi_rank_(inquire_environment(MPI_HOST))
    { }

    explicit rank(::yampi::io_process_t const)
      : mpi_rank_(inquire_environment(MPI_IO))
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

    bool operator==(rank const other) const { return mpi_rank_ == other.mpi_rank_; }
    bool operator<(rank const other) const { return mpi_rank_ < other.mpi_rank_; }

    int const& mpi_rank() const { return mpi_rank_; }

   private:
    int inquire_environment(int const key_value) const
    {
      // don't check flag because users cannnot delete the attribute MPI_HOST
      int* result;
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto flag = int{};
#   else
      auto flag = int();
#   endif
# else
      int flag;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code = MPI_Comm_get_attr(MPI_COMM_WORLD, key_value, &result, &flag);
# else
      int const error_code = MPI_Comm_get_attr(MPI_COMM_WORLD, key_value, &result, &flag);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::rank::inquire_environment"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rank::inquire_environment");
# endif

      return *result;
    }
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
  ::yampi::rank const any_source = ::yampi::rank{::yampi::any_source_t{}};
  ::yampi::rank const null_process = ::yampi::rank{::yampi::null_process_t{}};
#   else
  ::yampi::rank const any_source = ::yampi::rank(::yampi::any_source_t());
  ::yampi::rank const null_process = ::yampi::rank(::yampi::null_process_t());
#   endif
# endif

  inline bool exists_host_process()
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto const host = ::yampi::rank{::yampi::host_process_t{}};
#   else
    auto const host = ::yampi::rank(::yampi::host_process_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::rank const host = ::yampi::rank{::yampi::host_process_t{}};
#   else
    ::yampi::rank const host = ::yampi::rank(::yampi::host_process_t());
#   endif
# endif

    return host != ::yampi::null_process;
  }

  inline bool is_host_process(::yampi::rank self)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto const host = ::yampi::rank{::yampi::host_process_t{}};
#   else
    auto const host = ::yampi::rank(::yampi::host_process_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::rank const host = ::yampi::rank{::yampi::host_process_t{}};
#   else
    ::yampi::rank const host = ::yampi::rank(::yampi::host_process_t());
#   endif
# endif

    return self == host;
  }

  inline bool exists_io_process()
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto const io = ::yampi::rank{::yampi::io_process_t{}};
#   else
    auto const io = ::yampi::rank(::yampi::io_process_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::rank const io = ::yampi::rank{::yampi::io_process_t{}};
#   else
    ::yampi::rank const io = ::yampi::rank(::yampi::io_process_t());
#   endif
# endif

    return io != ::yampi::null_process;
  }

  inline bool is_io_process(::yampi::rank self)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto const io = ::yampi::rank{::yampi::io_process_t{}};
#   else
    auto const io = ::yampi::rank(::yampi::io_process_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::rank const io = ::yampi::rank{::yampi::io_process_t{}};
#   else
    ::yampi::rank const io = ::yampi::rank(::yampi::io_process_t());
#   endif
# endif

    return io == ::yampi::any_source or self == io;
  }
}


#endif

