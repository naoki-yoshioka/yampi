#ifndef YAMPI_ENVIRONMENT_HPP
# define YAMPI_ENVIRONMENT_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <stdexcept>
# include <string>

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

  class environment
  {
    static bool is_initialized_;
    static int error_code_on_last_finalize_;

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

    environment(int const argc, char const* argv[])
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
        auto const error_code = MPI_Init(const_cast<int*>(&argc), const_cast<char***>(&argv));
# else
        int const error_code = MPI_Init(const_cast<int*>(&argc), const_cast<char***>(&argv));
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
      if (!::yampi::is_finalized())
        error_code_on_last_finalize_ = MPI_Finalize();

      if (is_initialized_)
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

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::communicator world() const { return ::yampi::communicator{MPI_COMM_WORLD}; }
    ::yampi::communicator self() const { return ::yampi::communicator{MPI_COMM_SELF}; }
# else
    ::yampi::communicator world() const { return ::yampi::communicator(MPI_COMM_WORLD); }
    ::yampi::communicator self() const { return ::yampi::communicator(MPI_COMM_SELF); }
# endif

    std::string processor_name() const
    {
      char name[MPI_MAX_PROCESSOR_NAME];
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
        auto length = int{};
#   else
        auto length = int();
#   endif

      auto const error_code = MPI_Get_processor_name(name, &length);
# else
      int length;

      int const error_code = MPI_Get_processor_name(name, &length);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::environment::processor_name"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::environment::processor_name");
# endif

      return name;
    }

    double wall_clock_tick() const { return MPI_Wtick(); }
    double wall_clock_time() const { return MPI_Wtime(); }
  };

  bool environment::is_initialized_ = false;
  int environment::error_code_on_last_finalize_ = MPI_SUCCESS;

  inline bool is_finalized_successfully()
  { return environment::error_code_on_last_finalize() == MPI_SUCCESS; }
}


#endif

