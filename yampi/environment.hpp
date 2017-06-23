#ifndef YAMPI_ENVIRONMENT_HPP
# define YAMPI_ENVIRONMENT_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <stdexcept>

# include <mpi.h>

# include <yampi/communicator.hpp>
# include <yampi/is_initialized.hpp>
# include <yampi/is_finalized.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class already_initialized_error
    : public std::logic_error
  {
   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    already_initialized_error()
      : std::logic_error{"MPI environment has been already initialized"}
    { }
# else
    already_initialized_error()
      : std::logic_error("MPI environment has been already initialized")
    { }
# endif
  };

  template <typename T>
  class allocator;

  class datatype_committer;

  namespace {
    class environment
    {
      static bool is_initialized_;
      static int error_code_on_last_finalize_;

      static int num_unreleased_resources_;

      template <typename T>
      friend class ::yampi::allocator;
      friend class ::yampi::datatype_committer;

     public:
      BOOST_STATIC_CONSTEXPR int major_version = MPI_VERSION;
      BOOST_STATIC_CONSTEXPR int minor_version = MPI_SUBVERSION;

      environment()
      {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
        if (is_initialized_)
          throw ::yampi::already_initialized_error{};
# else
        if (is_initialized_)
          throw ::yampi::already_initialized_error();
# endif

        is_initialized_ = true;

        if (!::yampi::is_initialized())
        {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
          auto const error_code = MPI_Init(NULL, NULL);
# else
          int const error_code = MPI_Init(NULL, NULL);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
          if (error_code != MPI_SUCCESS)
            throw ::yampi::error{error_code, "yampi::environment::environment"};
# else
          if (error_code != MPI_SUCCESS)
            throw ::yampi::error(error_code, "yampi::environment::environment");
# endif
        }
      }

      environment(int argc, char* argv[])
      {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
        if (is_initialized_)
          throw ::yampi::already_initialized_error{};
# else
        if (is_initialized_)
          throw ::yampi::already_initialized_error();
# endif

        is_initialized_ = true;

        if (!::yampi::is_initialized())
        {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
          auto const error_code = MPI_Init(&argc, &argv);
# else
          int const error_code = MPI_Init(&argc, &argv);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
          if (error_code != MPI_SUCCESS)
            throw ::yampi::error{error_code, "yampi::environment::environment"};
# else
          if (error_code != MPI_SUCCESS)
            throw ::yampi::error(error_code, "yampi::environment::environment");
# endif
        }
      }

      ~environment() BOOST_NOEXCEPT_OR_NOTHROW
      {
        if (num_unreleased_resources_ == 0 and not ::yampi::is_finalized())
          error_code_on_last_finalize_ = MPI_Finalize();

        is_initialized_ = false;
      }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      environment(environment const&) = delete;
      environment& operator=(environment const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      environment(environment&&) = delete;
      environment& operator=(environment&&) = delete;
#   endif
# else
     private:
      environment(environment const&);
      environment& operator=(environment const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      environment(environment&&);
      environment& operator=(environment&&);
#   endif

     public:
# endif

      static int error_code_on_last_finalize() { return error_code_on_last_finalize_; }
    };

    bool environment::is_initialized_ = false;
    int environment::error_code_on_last_finalize_ = MPI_SUCCESS;

    int environment::num_unreleased_resources_ = 0;
  }

  inline bool is_finalized_successfully()
  { return environment::error_code_on_last_finalize() == MPI_SUCCESS; }
}


#endif

