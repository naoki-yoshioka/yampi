#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>


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

    explicit communicator(MPI_Comm const mpi_comm)
      : mpi_comm_(mpi_comm)
    { }

    explicit communicator(::yampi::world_communicator_t const)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit communicator(::yampi::self_communicator_t const)
      : mpi_comm_(MPI_COMM_SELF)
    { }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size", environment);
      return result;
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, &mpi_rank);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size", environment);
      return ::yampi::rank(mpi_rank);
    }

    void barrier(::yampi::environment const& environment) const
    {
      int const error_code = MPI_Barrier(mpi_comm_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::barrier", environment);
    }

    MPI_Comm const& mpi_comm() const { return mpi_comm_; }

    void swap(communicator& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Comm>::value ))
    {
      using std::swap;
      swap(mpi_comm_, other.mpi_comm_);
    }
  };

  inline void swap(::yampi::communicator& lhs, ::yampi::communicator& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::communicator >::value ))
  { lhs.swap(rhs); }


  inline ::yampi::communicator world_communicator()
  { return ::yampi::communicator(::yampi::world_communicator_t()); }

  inline ::yampi::communicator self_communicator()
  { return ::yampi::communicator(::yampi::self_communicator_t()); }


  inline bool is_valid_rank(
    ::yampi::rank const rank, ::yampi::communicator const communicator,
    ::yampi::environment const& environment)
  { return rank >= ::yampi::rank(0) and rank < ::yampi::rank(communicator.size(environment)); }
}


#endif

