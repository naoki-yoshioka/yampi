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
    communicator(communicator const&) = delete;
    communicator& operator=(communicator const&) = delete;
# else
   private:
    communicator();
    communicator(communicator const&);
    communicator& operator=(communicator const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    communicator(communicator&&) = default;
    communicator& operator=(communicator&&) = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    communicator(communicator&& other) : mpi_comm_(std::move(other.mpi_comm_)) { }
    communicator& operator=(communicator&& other)
    {
      if (this != YAMPI_addressof(other))
        mpi_comm_ = std::move(other.mpi_comm_);
      return *this;
    }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~communicator() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      MPI_Comm_free(YAMPI_addressof(mpi_comm_));
    }

    explicit communicator(MPI_Comm const mpi_comm)
      : mpi_comm_(mpi_comm)
    { }

    explicit communicator(::yampi::world_communicator_t const)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit communicator(::yampi::self_communicator_t const)
      : mpi_comm_(MPI_COMM_SELF)
    { }


    bool is_null() const { return mpi_comm_ == MPI_COMM_NULL; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size", environment);
      return result;
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, YAMPI_addressof(mpi_rank));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::size", environment);
      return ::yampi::rank(mpi_rank);
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


  inline bool is_valid_rank(
    ::yampi::rank const rank, ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  { return rank >= ::yampi::rank(0) and rank < ::yampi::rank(communicator.size(environment)); }
}


#endif

