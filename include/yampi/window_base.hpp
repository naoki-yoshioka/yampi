#ifndef YAMPI_WINDOW_BASE_HPP
# define YAMPI_WINDOW_BASE_HPP

# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <utility>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/group.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
# if MPI_VERSION >= 3
  enum class flavor
    : int
  {
    window = MPI_WIN_FLAVOR_CREATE, array = MPI_WIN_FLAVOR_ALLOCATE,
    dynamic = MPI_WIN_FLAVOR_DYNAMIC, array_shared = MPI_WIN_FLAVOR_SHARED
  };

  enum class memory_model
    : int
  { separate = MPI_WIN_SEPARATE, unified = MPI_WIN_UNIFIED };
# endif // MPI_VERSION >= 3

  template <typename Derived>
  class window_base
  {
   protected:
    MPI_Win mpi_win_;

   public:
    window_base() noexcept(std::is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_{MPI_WIN_NULL}
    { }

    window_base(window_base const&) = delete;
    window_base& operator=(window_base const&) = delete;

    window_base(window_base&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_{std::move(other.mpi_win_)}
    { other.mpi_win_ = MPI_WIN_NULL; }

    window_base& operator=(window_base&& other)
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

   protected:
    ~window_base() noexcept
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(std::addressof(mpi_win_));
    }

   public:
    explicit window_base(MPI_Win const& mpi_win)
      noexcept(std::is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_{mpi_win}
    { }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(std::addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::window_base::free", environment};
    }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Win const& mpi_win, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = mpi_win;
    }

    void reset(Derived&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = std::move(other.mpi_win_);
      other.mpi_win_ = MPI_WIN_NULL;
      derived().do_reset(std::move(other), environment);
    }

    bool is_null() const noexcept(noexcept(mpi_win_ == MPI_WIN_NULL)) { return mpi_win_ == MPI_WIN_NULL; }

    bool operator==(window_base const& other) const noexcept { return mpi_win_ == other.mpi_win_; }

    void group(::yampi::group& result, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Win_get_group(mpi_win_, std::addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::window_base::group", environment};
      result.reset(mpi_group, environment);
    }

    template <typename T>
    T* base_ptr(::yampi::environment const& environment) const
    {
      T* result;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_BASE, result, std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? result
        : throw ::yampi::error{error_code, "yampi::window_base::base_ptr", environment};
    }

    ::yampi::byte_displacement size_bytes(::yampi::environment const& environment) const
    {
      MPI_Aint result;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_SIZE, std::addressof(result), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? ::yampi::byte_displacement(result)
        : throw ::yampi::error{error_code, "yampi::window_base::size_bytes", environment};
    }

    int displacement_unit(::yampi::environment const& environment) const
    {
      int result;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_DISP_UNIT, std::addressof(result), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? result
        : throw ::yampi::error{error_code, "yampi::window_base::displacement_unit", environment};
    }

# if MPI_VERSION >= 3
    ::yampi::flavor flavor(::yampi::environment const& environment) const
    {
      int result;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_CREATE_FLAVOR, std::addressof(result), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::flavor>(result)
        : throw ::yampi::error{error_code, "yampi::window_base::flavor", environment};
    }

    ::yampi::memory_model memory_model(::yampi::environment const& environment) const
    {
      int result;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_MODEL, std::addressof(result), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::memory_model>(result)
        : throw ::yampi::error{error_code, "yampi::window_base::memory_model", environment};
    }

    void set_information(yampi::information const& information, yampi::environment const& environment) const
    {
      int const error_code = MPI_Win_set_info(mpi_win_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::window_base::set_information", environment};
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, std::addressof(result));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::window_base::get_information", environment};
      information.reset(result, environment);
    }
# endif

    MPI_Win const& mpi_win() const noexcept { return mpi_win_; }

    void swap(Derived& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
      derived().do_swap(other);
    }

   protected:
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    Derived const& derived() const noexcept { return static_cast<Derived const&>(*this); }
  };

  template <typename Derived>
  inline bool operator!=(::yampi::window_base<Derived> const& lhs, ::yampi::window_base<Derived> const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename Derived>
  inline void swap(::yampi::window_base<Derived>& lhs, ::yampi::window_base<Derived>& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

