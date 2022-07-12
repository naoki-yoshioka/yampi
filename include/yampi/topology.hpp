#ifndef YAMPI_TOPOLOGY_HPP
# define YAMPI_TOPOLOGY_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class topology
  {
   protected:
    ::yampi::communicator communicator_;

   public:
    topology() = delete;
    topology(topology const&) = delete;
    topology& operator=(topology const&) = delete;
    topology(topology&&) = default;
    topology& operator=(topology&&) = default;

   protected:
    ~topology() = default;

   public:
    explicit topology(MPI_Comm const& mpi_comm)
      noexcept(std::is_nothrow_copy_constructible<MPI_Comm>::value)
      : communicator_{mpi_comm}
    { }

    void reset(MPI_Comm const& mpi_comm, ::yampi::environment const& environment)
    { communicator_.reset(mpi_comm, environment); }

    void reset(topology&& other, ::yampi::environment const& environment)
    { communicator_.reset(std::move(other.communicator_), environment); }

    void free(::yampi::environment const& environment)
    { communicator_.free(environment); }

    ::yampi::communicator const& communicator() const noexcept { return communicator_; }

    void swap(topology& other) noexcept(YAMPI_is_nothrow_swappable< ::yampi::communicator >::value)
    {
      using std::swap;
      swap(communicator_, other.communicator_);
    }
  };

  inline void swap(::yampi::topology& lhs, ::yampi::topology& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

