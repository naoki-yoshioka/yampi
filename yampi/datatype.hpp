#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  class datatype
  {
    MPI_Datatype mpi_datatype_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    datatype() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_{MPI_DATATYPE_NULL}
    { }

    explicit datatype(MPI_Datatype const& mpi_datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_{mpi_datatype}
    { }
# else
    datatype() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }

    explicit datatype(MPI_Datatype const& mpi_datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(mpi_datatype)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    datatype(datatype const&) = default;
    datatype& operator=(datatype const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&&) = default;
    datatype& operator=(datatype&&) = default;
#   endif
    ~datatype() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    bool operator==(datatype const& other) const { return mpi_datatype_ == other.mpi_datatype_; }

    MPI_Datatype const& mpi_datatype() const { return mpi_datatype_; }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
  { return !(lhs == rhs); }
}


#endif

