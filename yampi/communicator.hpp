#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class environment;

  class communicator
  {
    MPI_Comm mpi_comm_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    BOOST_CONSTEXPR communicator() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_comm_{MPI_COMM_WORLD}
    { }
# else
    BOOST_CONSTEXPR communicator() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_comm_(MPI_COMM_WORLD)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    communicator(communicator const&) = default;
    communicator& operator=(communicator const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    communicator(communicator&&) = default;
    communicator& operator=(communicator&&) = default;
#   endif
    ~communicator() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    int size(::yampi::environment&) const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = int{};
#   else
      auto result = int();
#   endif

      auto const error_code = MPI_Comm_size(mpi_comm_, &result);
# else
      int result;

      int const error_code = MPI_Comm_size(mpi_comm_, &result);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::communicator::size"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size");
# endif

      return result;
    }

    ::yampi::rank rank(::yampi::environment&) const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto mpi_rank = int{};
#   else
      auto mpi_rank = int();
#   endif

      auto const error_code = MPI_Comm_size(mpi_comm_, &mpi_rank);
# else
      int mpi_rank;

      int const error_code = MPI_Comm_size(mpi_comm_, &mpi_rank);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::communicator::size"};

      return ::yampi::rank{mpi_rank};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size");

      return ::yampi::rank(mpi_rank);
# endif
    }

    MPI_Comm mpi_comm() const { return mpi_comm_; }
  };

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  auto BOOST_CONSTEXPR_OR_CONST world = ::yampi::communicator{};
#   else
  auto BOOST_CONSTEXPR_OR_CONST world = ::yampi::communicator();
#   endif
# else
  ::yampi::communicator BOOST_CONSTEXPR_OR_CONST world;
# endif
}


#endif

