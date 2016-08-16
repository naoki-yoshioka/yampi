#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  struct world_communicator_t { };
  struct self_communicator_t { };

  class communicator
  {
    MPI_Comm mpi_comm_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    communicator() = delete;
# else
   private:
    communicator();

   public:
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

# ifndef __FUJITSU
#   define YAMPI_CONSTEXPR BOOST_CONSTEXPR
# else
#   define YAMPI_CONSTEXPR
# endif
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    explicit BOOST_CONSTEXPR communicator(MPI_Comm const mpi_comm)
      : mpi_comm_{mpi_comm}
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::world_communicator_t const)
      : mpi_comm_{MPI_COMM_WORLD}
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::self_communicator_t const)
      : mpi_comm_{MPI_COMM_SELF}
    { }
# else
    explicit BOOST_CONSTEXPR communicator(MPI_Comm const mpi_comm)
      : mpi_comm_(mpi_comm)
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::world_communicator_t const)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::self_communicator_t const)
      : mpi_comm_(MPI_COMM_SELF)
    { }
# endif
# undef YAMPI_CONSTEXPR

    int size() const
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

    ::yampi::rank rank() const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto mpi_rank = int{};
#   else
      auto mpi_rank = int();
#   endif

      auto const error_code = MPI_Comm_rank(mpi_comm_, &mpi_rank);
# else
      int mpi_rank;

      int const error_code = MPI_Comm_rank(mpi_comm_, &mpi_rank);
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

    void barrier() const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code = MPI_Barrier(mpi_comm_);
# else
      int const error_code = MPI_Barrier(mpi_comm_);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::communicator::barrier"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::barrier");
# endif
    }

    MPI_Comm const& mpi_comm() const { return mpi_comm_; }
  };


# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  auto const world_communicator = ::yampi::communicator{::yampi::world_communicator_t{}};
  auto const self_communicator = ::yampi::communicator{::yampi::self_communicator_t{}};
#   else
  auto const world_communicator = ::yampi::communicator(::yampi::world_communicator_t());
  auto const self_communicator = ::yampi::communicator(::yampi::self_communicator_t());
#   endif
# else
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  ::yampi::communicator const world_communicator = ::yampi::communicator{::yampi::world_communicator_t{}};
  ::yampi::communicator const self_communicator = ::yampi::communicator{::yampi::self_communicator_t{}};
#   else
  ::yampi::communicator const world_communicator = ::yampi::communicator(::yampi::world_communicator_t());
  ::yampi::communicator const self_communicator = ::yampi::communicator(::yampi::self_communicator_t());
#   endif
# endif
}


#endif

