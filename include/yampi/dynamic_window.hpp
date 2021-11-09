#ifndef YAMPI_DYNAMIC_WINDOW_HPP
# define YAMPI_DYNAMIC_WINDOW_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>
# include <yampi/group.hpp>
# include <yampi/byte_displacement.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_SCOPED_ENUMS
#     define YAMPI_FLAVOR ::yampi::flavor
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model
#   else // BOOST_NO_CXX11_SCOPED_ENUMS
#     define YAMPI_FLAVOR ::yampi::flavor::flavor_
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model::memory_model_
#   endif // BOOST_NO_CXX11_SCOPED_ENUMS
# endif // MPI_VERSION >= 3


# if MPI_VERSION >= 3
namespace yampi
{
  class dynamic_window
    : public ::yampi::window_base< ::yampi::dynamic_window >
  {
    typedef ::yampi::window_base< ::yampi::dynamic_window > super_type;

    MPI_Win mpi_win_;

   public:
    dynamic_window()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_(MPI_WIN_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    dynamic_window(dynamic_window const&) = delete;
    dynamic_window& operator=(dynamic_window const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    dynamic_window(dynamic_window const&);
    dynamic_window& operator=(dynamic_window const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    dynamic_window(dynamic_window&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_(std::move(other.mpi_win_))
    { other.mpi_win_ = MPI_WIN_NULL; }

    dynamic_window& operator=(dynamic_window&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        if (mpi_win_ != MPI_WIN_NULL)
          MPI_Win_free(YAMPI_addressof(mpi_win_));
        mpi_win_ = std::move(other.mpi_win_);
        other.mpi_win_ = MPI_WIN_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~dynamic_window() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(YAMPI_addressof(mpi_win_));
    }

    dynamic_window(::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : mpi_win_(create(MPI_INFO_NULL, communicator, environment))
    { }

    dynamic_window(
      ::yampi::information const& information,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : mpi_win_(create(information.mpi_info(), communicator, environment))
    { }

   private:
    MPI_Win create(
      MPI_Info const& mpi_info,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Win result;
      int const error_code
        = MPI_Win_create_dynamic(mpi_info, communicator.mpi_comm(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::dynamic_window::create", environment);
    }

   public:
    bool operator==(dynamic_window const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_ == other.mpi_win_; }

    bool do_is_null() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_ == MPI_WIN_NULL; }

    MPI_Win const& do_mpi_win() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_; }

    template <typename T>
    T* do_base_ptr() const
    {
      T* base_ptr;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_BASE, base_ptr, YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? base_ptr
        : throw ::yampi::error(error_code, "yampi::window::do_base_ptr", environment);
    }

    ::yampi::byte_displacement do_size_bytes() const
    {
      MPI_Aint size_bytes;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_SIZE, YAMPI_addressof(size_bytes), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? ::yampi::byte_displacement(size_bytes)
        : throw ::yampi::error(error_code, "yampi::window::do_size_bytes", environment);
    }

    int do_displacement_unit() const
    {
      int displacement_unit;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_DISP_UNIT, YAMPI_addressof(displacement_unit), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? displacement_unit
        : throw ::yampi::error(error_code, "yampi::window::do_displacement_unit", environment);
    }

    YAMPI_FLAVOR do_flavor() const
    {
      int flavor;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_CREATE_FLAVOR, YAMPI_addressof(flavor), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<YAMPI_FLAVOR>(flavor)
        : throw ::yampi::error(error_code, "yampi::window::do_flavor", environment);
    }

    YAMPI_MEMORY_MODEL do_memory_model() const
    {
      int memory_model;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_MODEL, YAMPI_addressof(memory_model), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<YAMPI_MEMORY_MODEL>(memory_model)
        : throw ::yampi::error(error_code, "yampi::window::do_memory_model", environment);
    }

    void do_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Win_get_group(mpi_win_, YAMPI_addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::do_group", environment);
      group.reset(mpi_group, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(YAMPI_addressof(mpi_win_));
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

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(dynamic_window&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = std::move(other.mpi_win_);
      other.mpi_win_ = MPI_WIN_NULL;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(MPI_INFO_NULL, communicator, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(information.mpi_info(), communicator, environment);
    }

    void set_information(yampi::information const& information, yampi::environment const& environment) const
    {
      int const error_code = MPI_Win_set_info(mpi_win_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::dynamic_window::set_information", environment);
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? yampi::information(result)
        : throw ::yampi::error(error_code, "yampi::dynamic_window::get_information", environment);
    }

    void swap(dynamic_window& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
    }
  };

  inline bool operator!=(::yampi::dynamic_window const& lhs, ::yampi::dynamic_window const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::dynamic_window& lhs, ::yampi::dynamic_window& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}
# endif // MPI_VERSION >= 3


# if MPI_VERSION >= 3
#   undef YAMPI_MEMORY_MODEL
#   undef YAMPI_FLAVOR
# endif // MPI_VERSION >= 3
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_remove_cv

#endif

