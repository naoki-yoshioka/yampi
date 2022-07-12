#ifndef YAMPI_COMMUNICATOR_BASE_HPP
# define YAMPI_COMMUNICATOR_BASE_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/group.hpp>
# include <yampi/information.hpp>
# include <yampi/color.hpp>
# include <yampi/split_type.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class communicator_base
  {
   protected:
    MPI_Comm mpi_comm_;

   public:
    communicator_base() noexcept(std::is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(MPI_COMM_NULL)
    { }

    communicator_base(communicator_base const&) = delete;
    communicator_base& operator=(communicator_base const&) = delete;

    communicator_base(communicator_base&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Comm>::value
        and std::is_nothrow_copy_assignable<MPI_Comm>::value)
      : mpi_comm_(std::move(other.mpi_comm_))
    { other.mpi_comm_ = MPI_COMM_NULL; }

    communicator_base& operator=(communicator_base&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Comm>::value
        and std::is_nothrow_copy_assignable<MPI_Comm>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_comm_ != MPI_COMM_NULL and mpi_comm_ != MPI_COMM_WORLD and mpi_comm_ != MPI_COMM_SELF)
          MPI_Comm_free(std::addressof(mpi_comm_));
        mpi_comm_ = std::move(other.mpi_comm_);
        other.mpi_comm_ = MPI_COMM_NULL;
      }
      return *this;
    }

   protected:
    ~communicator_base() noexcept
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      MPI_Comm_free(std::addressof(mpi_comm_));
    }

   public:
    explicit communicator_base(MPI_Comm const& mpi_comm)
      noexcept(std::is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(mpi_comm)
    { }

    communicator_base(communicator_base const& other, ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, environment))
    { }

# if MPI_VERSION >= 3
    communicator_base(
      communicator_base const& other, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, information, environment))
    { }
# endif

    communicator_base(
      communicator_base const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
      : mpi_comm_(create(other, group, environment))
    { }

    communicator_base(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, color, key, environment))
    { }

# if MPI_VERSION >= 3
    communicator_base(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, information, environment))
    { }

    communicator_base(
      communicator_base const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, ::yampi::information(), environment))
    { }
# endif

   private:
    MPI_Comm duplicate(communicator_base const& other, ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code = MPI_Comm_dup(other.mpi_comm(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::duplicate", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm duplicate(
      communicator_base const& other, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_dup_with_info(other.mpi_comm(), information.mpi_info(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::duplicate", environment);
    }
# endif

    MPI_Comm create(
      communicator_base const& other, ::yampi::group const& group,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_create(other.mpi_comm(), group.mpi_group(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::create", environment);
    }

    MPI_Comm split(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split(other.mpi_comm(), color.mpi_color(), key, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::split", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm split(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split_type(
            other.mpi_comm(), split_type.mpi_split_type(), key, information.mpi_info(),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::split", environment);
    }
# endif

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Comm const& mpi_comm, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = mpi_comm;
    }

    void reset(communicator_base&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = std::move(other.mpi_comm_);
      other.mpi_comm_ = MPI_COMM_NULL;
    }

    void reset(communicator_base const& other, ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = duplicate(other, environment);
    }

    void reset(communicator_base const& other, ::yampi::group const& group, ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = create(other, group, environment);
    }

    void reset(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, color, key, environment);
    }

# if MPI_VERSION >= 3
    void reset(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, split_type, key, information, environment);
    }

    void reset(
      communicator_base const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, split_type, key, ::yampi::information(), environment);
    }
# endif

    void free(::yampi::environment const& environment)
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      int const error_code = MPI_Comm_free(std::addressof(mpi_comm_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::free", environment);
    }

    bool is_null() const noexcept(noexcept(mpi_comm_ == MPI_COMM_NULL))
    { return mpi_comm_ == MPI_COMM_NULL; }

    bool operator==(communicator_base const& other) const noexcept
    { return mpi_comm_ == other.mpi_comm_; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::size", environment);
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, std::addressof(mpi_rank));
      return error_code == MPI_SUCCESS
        ? ::yampi::rank(mpi_rank)
        : throw ::yampi::error(error_code, "yampi::communicator_base::rank", environment);
    }

    void group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Comm_group(mpi_comm_, std::addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::group", environment);

      group.reset(mpi_group, environment);
    }

# if MPI_VERSION >= 3
    void set_information(::yampi::information const& information, ::yampi::environment const& environment) const
    {
      int const error_code = MPI_Comm_set_info(mpi_comm_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::set_information", environment);
    }

    void get_information(::yampi::information& information, ::yampi::environment const& environment) const
    {
      MPI_Info mpi_info;
      int const error_code = MPI_Comm_get_info(mpi_comm_, std::addressof(mpi_info));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::get_information", environment);
      information.reset(mpi_info, environment);
    }
# endif

    bool is_intercommunicator(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_test_inter(mpi_comm_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? static_cast<bool>(result)
        : throw ::yampi::error(error_code, "yampi::communicator_base::is_intercommunicator", environment);
    }

    MPI_Comm const& mpi_comm() const noexcept { return mpi_comm_; }

    void swap(communicator_base& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Comm>::value)
    {
      using std::swap;
      swap(mpi_comm_, other.mpi_comm_);
    }
  }; // class communicator_base

  inline void swap(::yampi::communicator_base& lhs, ::yampi::communicator_base& rhs)
    noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  inline bool is_valid_rank(
    ::yampi::rank const& rank, ::yampi::communicator_base const& communicator,
    ::yampi::environment const& environment)
  {
    return rank >= ::yampi::rank(0)
      and rank < ::yampi::rank(communicator.size(environment));
  }
} // namespace yampi


# undef YAMPI_is_nothrow_swappable

#endif // YAMPI_COMMUNICATOR_BASE_HPP
