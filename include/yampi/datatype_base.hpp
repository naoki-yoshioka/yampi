#ifndef YAMPI_DATATYPE_BASE_HPP
# define YAMPI_DATATYPE_BASE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/bounds.hpp>

namespace yampi
{
  template <typename Derived>
  class datatype_base
  {
   public:
# if MPI_VERSION >= 3
    typedef MPI_Count size_type;
    typedef MPI_Count count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# else
    typedef int size_type;
    typedef MPI_Aint count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# endif

    bool is_null() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_is_null(); }

    MPI_Datatype mpi_datatype() const BOOST_NOEXCEPT_OR_NOTHROW { return derived().do_mpi_datatype(); }

   protected:
    Derived& derived() BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived&>(*this); }
    Derived const& derived() const BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<Derived const&>(*this); }
  }; // class datatype_base<Derived>
} // namespace yampi

#endif // YAMPI_DATATYPE_BASE_HPP
