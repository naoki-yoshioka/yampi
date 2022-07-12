#ifndef YAMPI_WINDOW_HPP
# define YAMPI_WINDOW_HPP

# include <cassert>
# include <cstddef>
# include <iterator>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>
# include <yampi/group.hpp>
# include <yampi/byte_displacement.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class window
    : public ::yampi::window_base< ::yampi::window >
  {
    typedef ::yampi::window_base< ::yampi::window > super_type;

    MPI_Win mpi_win_;

   public:
    window() noexcept(std::is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_(MPI_WIN_NULL)
    { }

    window(window const&) = delete;
    window& operator=(window const&) = delete;

    window(window&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_(std::move(other.mpi_win_))
    { other.mpi_win_ = MPI_WIN_NULL; }

    window& operator=(window&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_win_ != MPI_WIN_NULL)
          MPI_Win_free(std::addressof(mpi_win_));
        mpi_win_ = std::move(other.mpi_win_);
        other.mpi_win_ = MPI_WIN_NULL;
      }
      return *this;
    }

    ~window() noexcept
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(std::addressof(mpi_win_));
    }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, MPI_INFO_NULL, communicator, environment))
    { assert(last >= first); }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, information.mpi_info(), communicator, environment))
    { assert(last >= first); }

   private:
    template <typename ContiguousIterator>
    MPI_Win create(
      ContiguousIterator const first, ContiguousIterator const last,
      MPI_Info const& mpi_info, ::yampi::communicator const& communicator,
      ::yampi::environment const& environment) const
    {
      assert(last >= first);

      using value_type = typename std::remove_cv<typename std::iterator_traits<ContiguousIterator>::value_type>::type;
      MPI_Win result;
      int const error_code
        = MPI_Win_create(
            std::addressof(*first),
            (::yampi::addressof(*last, environment) - ::yampi::addressof(*first, environment)).mpi_byte_displacement(),
            sizeof(value_type), mpi_info, communicator.mpi_comm(),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::window::create", environment);
    }

   public:
    bool operator==(window const& other) const noexcept { return mpi_win_ == other.mpi_win_; }

    bool do_is_null() const noexcept { return mpi_win_ == MPI_WIN_NULL; }

    MPI_Win const& do_mpi_win() const noexcept { return mpi_win_; }

    template <typename T>
    T* do_base_ptr(::yampi::environment const& environment) const
    {
      T* base_ptr;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_BASE, base_ptr, std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? base_ptr
        : throw ::yampi::error(error_code, "yampi::window::do_base_ptr", environment);
    }

    ::yampi::byte_displacement do_size_bytes(::yampi::environment const& environment) const
    {
      MPI_Aint size_bytes;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_SIZE, std::addressof(size_bytes), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? ::yampi::byte_displacement(size_bytes)
        : throw ::yampi::error(error_code, "yampi::window::do_size_bytes", environment);
    }

    int do_displacement_unit(::yampi::environment const& environment) const
    {
      int displacement_unit;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_DISP_UNIT, std::addressof(displacement_unit), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? displacement_unit
        : throw ::yampi::error(error_code, "yampi::window::do_displacement_unit", environment);
    }

# if MPI_VERSION >= 3
    ::yampi::flavor do_flavor(::yampi::environment const& environment) const
    {
      int flavor;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_CREATE_FLAVOR, std::addressof(flavor), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::flavor>(flavor)
        : throw ::yampi::error(error_code, "yampi::window::do_flavor", environment);
    }

    ::yampi::memory_model do_memory_model(::yampi::environment const& environment) const
    {
      int memory_model;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_MODEL, std::addressof(memory_model), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::memory_model>(memory_model)
        : throw ::yampi::error(error_code, "yampi::window::do_memory_model", environment);
    }
# endif // MPI_VERSION >= 3

    void do_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Win_get_group(mpi_win_, std::addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::do_group", environment);
      group.reset(mpi_group, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(std::addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::free", environment);
    }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Win const& mpi_win, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = mpi_win;
    }

    void reset(window&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = std::move(other.mpi_win_);
      other.mpi_win_ = MPI_WIN_NULL;
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, MPI_INFO_NULL, communicator, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, information.mpi_info(), communicator, environment);
    }

    void set_information(yampi::information const& information, yampi::environment const& environment) const
    {
      int const error_code = MPI_Win_set_info(mpi_win_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::set_information", environment);
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, std::addressof(result));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::get_information", environment);
      information.reset(result, environment);
    }

    void swap(window& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
    }
  };

  inline bool operator!=(::yampi::window const& lhs, ::yampi::window const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::window& lhs, ::yampi::window& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

