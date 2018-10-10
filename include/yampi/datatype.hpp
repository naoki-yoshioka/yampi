#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/address.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class datatype
  {
    MPI_Datatype mpi_datatype_;

   public:
    datatype() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }

    explicit datatype(MPI_Datatype const& mpi_datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(mpi_datatype)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    datatype(datatype const&) = default;
    datatype& operator=(datatype const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&&) = default;
    datatype& operator=(datatype&&) = default;
#   endif
    ~datatype() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    // TODO: implement constructor using MPI_Type_create_resized

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Type_size(mpi_datatype_, YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::size", environment);
      return result;
    }

    std::pair< ::yampi::address, ::yampi::address >
    lower_bound_extent(::yampi::environment const& environment) const
    {
      MPI_Aint lower_bound, extent;
      int const error_code
        = MPI_Type_get_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "yampi::datatype::lower_bound_extent", environment);
      return std::make_pair(::yampi::address(lower_bound), ::yampi::address(extent));
    }

    bool is_null() const { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    bool operator==(datatype const& other) const { return mpi_datatype_ == other.mpi_datatype_; }

    MPI_Datatype const& mpi_datatype() const { return mpi_datatype_; }

    void swap(datatype& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs)
    BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable< ::yampi::datatype >::value)
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof

#endif

