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
    explicit BOOST_CONSTEXPR communicator(MPI_Comm const mpi_comm)
      : mpi_comm_(mpi_comm)
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::world_communicator_t const)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit YAMPI_CONSTEXPR communicator(::yampi::self_communicator_t const)
      : mpi_comm_(MPI_COMM_SELF)
    { }
# undef YAMPI_CONSTEXPR

    int size() const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size");
      return result;
    }

    ::yampi::rank rank() const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, &mpi_rank);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size");
      return ::yampi::rank(mpi_rank);
    }

    void barrier() const
    {
      int const error_code = MPI_Barrier(mpi_comm_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::barrier");
    }

    MPI_Comm const& mpi_comm() const { return mpi_comm_; }
  };

  inline bool operator==(::yampi::communicator const lhs, ::yampi::communicator const rhs)
  {
    int result;
    int const error_code = MPI_Comm_compare(lhs.mpi_comm(), rhs.mpi_comm(), &result);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::communicaotr::operator==");
    return result == MPI_IDENT;
  }


  ::yampi::communicator const world_communicator
    = ::yampi::communicator(::yampi::world_communicator_t());
  ::yampi::communicator const self_communicator
    = ::yampi::communicator(::yampi::self_communicator_t());
}


#endif

