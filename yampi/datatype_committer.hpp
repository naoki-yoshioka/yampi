#ifndef YAMPI_DATATYPE_COMMITTER_HPP
# define YAMPI_DATATYPE_COMMITTER_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/error.hpp>
# include <yampi/datatype.hpp>
# include <yampi/is_initialized.hpp>
# include <yampi/environment.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class datatype_committer
  {
    ::yampi::datatype datatype_;
    static int error_code_on_last_freeing_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    datatype_committer(::yampi::datatype const& datatype)
      : datatype_{datatype}
    {
      assert(::yampi::is_initialized());

#   ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto error_code = MPI_Type_commit(const_cast<MPI_Datatype*>(YAMPI_addressof(datatype.mpi_datatype())));
#   else
      int error_code = MPI_Type_commit(const_cast<MPI_Datatype*>(YAMPI_addressof(datatype.mpi_datatype())));
#   endif

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::datatype_committer"};

      ++::yampi::environment::num_unreleased_resources_;
    }
# else
    datatype_committer(::yampi::datatype const& datatype)
      : datatype_(datatype)
    {
#   ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto error_code = MPI_Type_commit(const_cast<MPI_Datatype*>(YAMPI_addressof(datatype.mpi_datatype())));
#   else
      int error_code = MPI_Type_commit(const_cast<MPI_Datatype*>(YAMPI_addressof(datatype.mpi_datatype())));
#   endif

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype_committer");

      ++::yampi::environment::num_unreleased_resources_;
    }
# endif

    ~datatype_committer() BOOST_NOEXCEPT_OR_NOTHROW
    {
      error_code_on_last_freeing_ = MPI_Type_free(const_cast<MPI_Datatype*>(YAMPI_addressof(datatype_.mpi_datatype())));

      --::yampi::environment::num_unreleased_resources_;
      if (::yampi::environment::num_unreleased_resources_ == 0
          and not ::yampi::environment::is_initialized_)
      {
        if (not ::yampi::is_finalized())
          ::yampi::environment::error_code_on_last_finalize_ = MPI_Finalize();

        ::yampi::environment::is_initialized_ = false;
      }
    }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    datatype_committer(datatype_committer const&) = delete;
    datatype_committer& operator=(datatype_committer const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype_committer(datatype_committer&&) = delete;
    datatype_committer& operator=(datatype_committer&&) = delete;
#   endif
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    datatype_committer(datatype_committer const&);
    datatype_committer& operator=(datatype_committer const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype_committer(datatype_committer&&);
    datatype_committer& operator=(datatype_committer&&);
#   endif

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

    static int error_code_on_last_freeing() { return error_code_on_last_freeing_; }

    bool operator==(datatype_committer const& other) const { return datatype_ == other.datatype_; }

    ::yampi::datatype const& datatype() const { return datatype_; }
  };

  int datatype_committer::error_code_on_last_freeing_ = MPI_SUCCESS;

  inline bool operator!=(::yampi::datatype_committer const& lhs, ::yampi::datatype_committer const& rhs)
  { return !(lhs == rhs); }
}


# undef YAMPI_addressof

#endif

