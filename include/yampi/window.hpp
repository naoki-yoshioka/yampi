#ifndef PASTEL_WINDOW_HPP
# define PASTEL_WINDOW_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/communicator.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class window
  {
    MPI_Win mpi_win_;

   public:
    window() : mpi_win_(MPI_WIN_NULL) { }
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    window(window const&) = delete;
    window& operator=(window const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    window(window const&);
    window& operator=(window const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    window(window&& other)
      : mpi_win_(std::move(other.mpi_win_))
    { other.mpi_win_ = MPI_WIN_NULL; }

    window& operator=(window&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_win_ = std::move(other.mpi_win_);
        other.mpi_win_ = MPI_WIN_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~window() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(YAMPI_addressof(mpi_win_));
    }

    explicit window(MPI_Win const mpi_win)
      : mpi_win_(mpi_win)
    { }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, MPI_INFO_NULL, communicator, environment))
    { }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, information.mpi_info(), communicator, environment))
    { }

    // TODO: implement other constructors

   private:
    template <typename ContiguousIterator>
    MPI_Win create(
      ContiguousIterator const first, ContiguousIterator const last,
      MPI_Info mpi_info, ::yampi::communicator const& communicator,
      ::yampi::environment const& environment) const
    {
      typedef std::iterator_traits<ContiguousIterator>::value_type value_type;
      MPI_Win result;
      int const error_code
        = MPI_Win_create(
            YAMPI_addressof(*first),
            (::yampi::addressof(*last) - ::yampi::addressof(*first)).mpi_address(),
            sizeof(value_type), mpi_info, communicator.mpi_comm(),
            YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::create", environment);

      return result;
    }

   public:
    void release(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(YAMPI_addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::release", environment);
    }

    // TODO: implement attributes and info

    MPI_Win const& mpi_win() const { return mpi_win_; }
  };
}


# undef YAMPI_addressof

#endif

