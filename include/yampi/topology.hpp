#ifndef YAMPI_TOPOLOGY_HPP
# define YAMPI_TOPOLOGY_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class topology
  {
   protected:
    ::yampi::communicator communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    topology() = delete;
    topology(topology const&) = delete;
    topology& operator=(topology const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    topology();
    topology(topology const&);
    topology& operator=(topology const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    topology(topology&&) = default;
    topology& operator=(topology&&) = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    topology(topology&& other) : communicator_(std::move(other.communicator_)) { }
    topology& operator=(topology&& other)
    {
      if (this != YAMPI_addressof(other))
        communicator_ = std::move(other.communicator_);
      return *this;
    }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

   protected:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~topology() = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~topology() { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

   public:
    explicit topology(MPI_Comm const mpi_comm)
      : communicator_(mpi_comm)
    { }

   public:
    void reset(MPI_Comm const mpi_comm, ::yampi::environment const& environment)
    { communicator_.reset(mpi_comm, environment); }

    void free(::yampi::environment const& environment)
    { communicator_.free(environment); }


    ::yampi::communicator const& communicator() const { return communicator_; }
  };
}


# undef YAMPI_addressof

#endif

