#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/group.hpp>
# include <yampi/tag.hpp>
# include <yampi/color.hpp>
# include <yampi/split_type.hpp>


namespace yampi
{
  struct world_communicator_t { };
  struct self_communicator_t { };

  namespace tags
  {
# if __cplusplus >= 201703L
    inline constexpr ::yampi::world_communicator_t world_communicator{};
    inline constexpr ::yampi::self_communicator_t self_communicator{};
# else
    constexpr ::yampi::world_communicator_t world_communicator{};
    constexpr ::yampi::self_communicator_t self_communicator{};
# endif
  }

  class communicator
    : public ::yampi::communicator_base
  {
    typedef ::yampi::communicator_base base_type;

   public:
    communicator() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    communicator(communicator const&) = delete;
    communicator& operator=(communicator const&) = delete;
    communicator(communicator&&) = default;
    communicator& operator=(communicator&&) = default;
    ~communicator() noexcept = default;

    using base_type::base_type;

    explicit communicator(::yampi::world_communicator_t const)
      noexcept(std::is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type{MPI_COMM_WORLD}
    { }

    explicit communicator(::yampi::self_communicator_t const)
      noexcept(std::is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type{MPI_COMM_SELF}
    { }

# if MPI_VERSION >= 3
    // only for intracommunicator
    communicator(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment)
      : base_type{create(other, group, tag, environment)}
    { }
# endif

   private:
# if MPI_VERSION >= 3
    // only for intracommunicator
    MPI_Comm create(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_create_group(
            other.mpi_comm(), group.mpi_group(), tag.mpi_tag(),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::create", environment);
    }
# endif

   public:
    using base_type::reset;

    void reset(yampi::world_communicator_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = MPI_COMM_WORLD;
    }

    void reset(yampi::self_communicator_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = MPI_COMM_SELF;
    }

# if MPI_VERSION >= 3
    // only for intracommunicator
    void reset(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment)
    {
      if (this == std::addressof(other))
        return;

      free(environment);
      mpi_comm_ = create(other, group, tag, environment);
    }
# endif
  };

  inline void swap(::yampi::communicator& lhs, ::yampi::communicator& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

# if __cplusplus >= 201703L
  inline ::yampi::communicator world_communicator{::yampi::tags::world_communicator};
  inline ::yampi::communicator self_communicator{::yampi::tags::self_communicator};
# endif
}


#endif

