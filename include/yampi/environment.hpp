#ifndef YAMPI_ENVIRONMENT_HPP
# define YAMPI_ENVIRONMENT_HPP

# include <cstddef>
# include <stdexcept>

# include <mpi.h>

# include <yampi/is_initialized.hpp>
# include <yampi/is_finalized.hpp>
# include <yampi/error.hpp>
# include <yampi/thread_support.hpp>


namespace yampi
{
  class initialization_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit initialization_error(int const error_code)
      : std::runtime_error{"Error occurred when initializing"},
        error_code_{error_code}
    { }

    int error_code() const { return error_code_; }
  };

  class already_initialized_error
    : public std::logic_error
  {
   public:
    already_initialized_error()
      : std::logic_error{"MPI environment has been already initialized"}
    { }
  };

  class already_finalized_error
    : public std::logic_error
  {
   public:
    already_finalized_error()
      : std::logic_error{"MPI environment has been already finalized"}
    { }
  };


  class environment
  {
    ::yampi::thread_support thread_support_;

   public:
    static constexpr int major_version = MPI_VERSION;
    static constexpr int minor_version = MPI_SUBVERSION;

    environment()
      : thread_support_{::yampi::thread_support::single}
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int const error_code = MPI_Init(NULL, NULL);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);
    }

    environment(int argc, char* argv[])
      : thread_support_{::yampi::thread_support::single}
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int const error_code = MPI_Init(&argc, &argv);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);
    }

    explicit environment(::yampi::thread_support const thread_support)
      : thread_support_{}
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int provided_thread_support;
      int const error_code
        = MPI_Init_thread(
            NULL, NULL,
            static_cast<int>(thread_support), &provided_thread_support);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);

      thread_support_ = static_cast< ::yampi::thread_support >(provided_thread_support);
    }

    environment(int argc, char* argv[], ::yampi::thread_support const thread_support)
      : thread_support_{}
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int provided_thread_support;
      int const error_code
        = MPI_Init_thread(
            &argc, &argv,
            static_cast<int>(thread_support), &provided_thread_support);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);

      thread_support_ = static_cast< ::yampi::thread_support >(provided_thread_support);
    }

    ~environment() noexcept
    {
      int is_finalized;
      int const error_code = MPI_Finalized(&is_finalized);
      if (error_code != MPI_SUCCESS or static_cast<bool>(is_finalized))
        return;

      MPI_Finalize();
    }

    environment(environment const&) = delete;
    environment& operator=(environment const&) = delete;
    environment(environment&&) = delete;
    environment& operator=(environment&&) = delete;

    void finalize()
    {
      if (::yampi::is_finalized())
        throw ::yampi::already_finalized_error();

      int const error_code = MPI_Finalize();
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::environment::finalize", *this);
    }


    ::yampi::thread_support thread_support() const noexcept { return thread_support_; }

    ::yampi::thread_support query_thread_support() const
    {
      int provided_thread_support;
      int const error_code = MPI_Query_thread(&provided_thread_support);
      return error_code == MPI_SUCCESS
        ? static_cast< ::yampi::thread_support >(provided_thread_support)
        : throw ::yampi::error(error_code, "yampi::environment::query_thread_support", *this);
    }

    bool is_main_thread() const
    {
      int flag;
      int const error_code = MPI_Is_thread_main(&flag);
      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::environment::is_main_thread", *this);
    }
  };
}


#endif

