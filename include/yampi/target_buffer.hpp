#ifndef YAMPI_TARGET_BUFFER_HPP
# define YAMPI_TARGET_BUFFER_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  template <typename T, typename Enable = void>
  class target_buffer
  {
    MPI_Aint mpi_displacement_;
    int count_;
    ::yampi::datatype const* datatype_ptr_;

   public:
    template <typename Integer>
    target_buffer(Integer const displacement, ::yampi::datatype const& datatype) noexcept
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(1),
        datatype_ptr_(std::addressof(datatype))
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
    }

    template <typename Integer>
    target_buffer(Integer const displacement, int const count, ::yampi::datatype const& datatype) noexcept
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(count),
        datatype_ptr_(std::addressof(datatype))
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
      assert(count >= 0);
    }

    bool operator==(target_buffer const& other) const noexcept(noexcept(*datatype_ptr_ == *(other.datatype_ptr_)))
    { return mpi_displacement_ == other.mpi_displacement_ and count_ == other.count_ and *datatype_ptr_ == *(other.datatype_ptr_); }

    MPI_Aint const& mpi_displacement() const noexcept { return mpi_displacement_; }
    int const& count() const noexcept { return count_; }
    ::yampi::datatype const& datatype() const noexcept { return *datatype_ptr_; }

    void swap(target_buffer& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_displacement_, other.mpi_displacement_);
      swap(count_, other.count_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class target_buffer<T, Enable>

  template <typename T>
  class target_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    MPI_Aint mpi_displacement_;
    int count_;

   public:
    template <typename Integer>
    explicit target_buffer(Integer const displacement) noexcept
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(1)
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
    }

    template <typename Integer>
    target_buffer(Integer const displacement, int const count) noexcept
      : mpi_displacement_(static_cast<MPI_Aint>(displacement)), count_(count)
    {
      static_assert(std::is_integral<Integer>::value, "Integer should be an integral type");
      assert(displacement >= Integer{0});
      assert(count >= 0);
    }

    bool operator==(target_buffer const& other) const noexcept
    { return mpi_displacement_ == other.mpi_displacement_ and count_ == other.count_; }

    MPI_Aint const& mpi_displacement() const noexcept { return mpi_displacement_; }
    int const& count() const noexcept { return count_; }
    ::yampi::predefined_datatype<T> datatype() const noexcept { return ::yampi::predefined_datatype<T>(); }

    void swap(target_buffer& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_displacement_, other.mpi_displacement_);
      swap(count_, other.count_);
    }
  }; // class target_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  inline bool operator!=(::yampi::target_buffer<T> const& lhs, ::yampi::target_buffer<T> const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::target_buffer<T>& lhs, ::yampi::target_buffer<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename T, typename Integer>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(Integer const displacement)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, datatype)))
  { return ::yampi::target_buffer<T>(displacement, datatype); }

  template <typename T, typename Integer>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(Integer const displacement, int count)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, int count, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T, typename Integer>
  inline ::yampi::target_buffer<T> make_target_buffer(Integer const displacement, int count, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count, datatype)))
  { return ::yampi::target_buffer<T>(displacement, count, datatype); }
}


# undef YAMPI_is_nothrow_swappable

#endif
