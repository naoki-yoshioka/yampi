#ifndef YAMPI_WINDOW_BASE_HPP
# define YAMPI_WINDOW_BASE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/group.hpp>


namespace yampi
{
# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class flavor
    : int
  {
    window = MPI_WIN_FLAVOR_CREATE, array = MPI_WIN_FLAVOR_ALLOCATE,
    dynamic = MPI_WIN_FLAVOR_DYNAMIC, array_shared = MPI_WIN_FLAVOR_SHARED
  };

  enum class memory_model
    : int
  { separate = MPI_WIN_SEPARATE, unified = MPI_WIN_UNIFIED };

#     define YAMPI_FLAVOR ::yampi::flavor
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model
#   else // BOOST_NO_CXX11_SCOPED_ENUMS
  namespace flavor
  {
    enum flavor_
    {
      window = MPI_WIN_FLAVOR_CREATE, array = MPI_WIN_FLAVOR_ALLOCATE,
      dynamic = MPI_WIN_FLAVOR_DYNAMIC, array_shared = MPI_WIN_FLAVOR_SHARED
    };
  }

  namespace memory_model
  {
    enum memory_model_
    { separate = MPI_WIN_SEPARATE, unified = MPI_WIN_UNIFIED };
  }

#     define YAMPI_FLAVOR ::yampi::flavor::flavor_
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model::memory_model_
#   endif // BOOST_NO_CXX11_SCOPED_ENUMS
# endif // MPI_VERSION >= 3

  template <typename Derived>
  class window_base
  {
   public:
    bool is_null() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_is_null(); }

    MPI_Win const& mpi_win() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_mpi_win(); }

    template <typename T>
    T* base_ptr() const { return derived().do_base_ptr<T>(); }
    ::yampi::byte_displacement size_bytes() const { return derived().do_size_bytes(); }
    int displacement_unit() const { return derived().do_displacement_unit(); }
# if MPI_VERSION >= 3
    YAMPI_FLAVOR flavor() const { return derived().do_flavor(); }
    YAMPI_MEMORY_MODEL memory_model() const { return derived().do_memory_model(); }
#   undef YAMPI_MEMORY_MODEL
#   undef YAMPI_FLAVOR
# endif // MPI_VERSION >= 3

    void group(::yampi::group& group, ::yampi::environment const& environment) const { derived().do_group(group, environment); }

   protected:
    Derived& derived() BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived&>(*this); }
    Derived const& derived() const BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived const&>(*this); }
  };
}


#endif

