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
# include <yampi/displacement.hpp>
# if MPI_VERSION >= 4
#   include <yampi/count.hpp>
# endif // MPI_VERSION >= 4

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
    ::yampi::displacement displacement_;
# if MPI_VERSION >= 4
    ::yampi::count count_;
# else // MPI_VERSION >= 4
    int count_;
# endif // MPI_VERSION >= 4
    ::yampi::datatype const* datatype_ptr_;

   public:
    target_buffer(::yampi::displacement const displacement, ::yampi::datatype const& datatype) noexcept
      : displacement_{displacement}, count_{1},
        datatype_ptr_{std::addressof(datatype)}
    { assert(displacement >= ::yampi::displacement{MPI_Aint{0}}); }

# if MPI_VERSION >= 4
    target_buffer(::yampi::displacement const displacement, ::yampi::count const count, ::yampi::datatype const& datatype) noexcept
      : displacement_{displacement}, count_{count},
        datatype_ptr_{std::addressof(datatype)}
    {
      assert(displacement >= ::yampi::displacement{MPI_Aint{0}});
      assert(count >= ::yampi::count{0});
    }
# else // MPI_VERSION >= 4
    target_buffer(::yampi::displacement const displacement, int const count, ::yampi::datatype const& datatype) noexcept
      : displacement_{displacement}, count_{count},
        datatype_ptr_{std::addressof(datatype)}
    {
      assert(displacement >= ::yampi::displacement{MPI_Aint{0}});
      assert(count >= 0);
    }
# endif // MPI_VERSION >= 4

    bool operator==(target_buffer const& other) const noexcept(noexcept(*datatype_ptr_ == *(other.datatype_ptr_)))
    { return displacement_ == other.displacement_ and count_ == other.count_ and *datatype_ptr_ == *(other.datatype_ptr_); }

    ::yampi::displacement const& displacement() const noexcept { return displacement_; }
# if MPI_VERSION >= 4
    ::yampi::count const& count() const noexcept { return count_; }
# else // MPI_VERSION >= 4
    int const& count() const noexcept { return count_; }
# endif // MPI_VERSION >= 4
    ::yampi::datatype const& datatype() const noexcept { return *datatype_ptr_; }

    void swap(target_buffer& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(displacement_, other.displacement_);
      swap(count_, other.count_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class target_buffer<T, Enable>

  template <typename T>
  class target_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    ::yampi::displacement displacement_;
# if MPI_VERSION >= 4
    ::yampi::count count_;
# else // MPI_VERSION >= 4
    int count_;
# endif // MPI_VERSION >= 4

   public:
    explicit target_buffer(::yampi::displacement const displacement) noexcept
      : displacement_{displacement}, count_{1}
    { assert(displacement >= ::yampi::displacement{MPI_Aint{0}}); }

# if MPI_VERSION >= 4
    target_buffer(::yampi::displacement const displacement, ::yampi::count const count) noexcept
      : displacement_{displacement}, count_{count}
    {
      assert(displacement >= ::yampi::displacement{MPI_Aint{0}});
      assert(count >= ::yampi::count{0});
    }
# else // MPI_VERSION >= 4
    target_buffer(::yampi::displacement const displacement, int const count) noexcept
      : displacement_{displacement}, count_{count}
    {
      assert(displacement >= ::yampi::displacement{MPI_Aint{0}});
      assert(count >= 0);
    }
# endif // MPI_VERSION >= 4

    bool operator==(target_buffer const& other) const noexcept
    { return displacement_ == other.displacement_ and count_ == other.count_; }

    ::yampi::displacement const& displacement() const noexcept { return displacement_; }
# if MPI_VERSION >= 4
    ::yampi::count const& count() const noexcept { return count_; }
# else // MPI_VERSION >= 4
    int const& count() const noexcept { return count_; }
# endif // MPI_VERSION >= 4
    ::yampi::predefined_datatype<T> datatype() const noexcept { return ::yampi::predefined_datatype<T>(); }

    void swap(target_buffer& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(displacement_, other.displacement_);
      swap(count_, other.count_);
    }
  }; // class target_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  inline bool operator!=(::yampi::target_buffer<T> const& lhs, ::yampi::target_buffer<T> const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::target_buffer<T>& lhs, ::yampi::target_buffer<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename T>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(::yampi::displacement const displacement)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T>
  inline ::yampi::target_buffer<T> make_target_buffer(::yampi::displacement const displacement, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement)))
  { return ::yampi::target_buffer<T>(displacement); }

  template <typename T>
  inline ::yampi::target_buffer<T> make_target_buffer(::yampi::displacement const displacement, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, datatype)))
  { return ::yampi::target_buffer<T>(displacement, datatype); }

  template <typename T>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::target_buffer<T> >::type make_target_buffer(::yampi::displacement const displacement, int count)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T>
  inline ::yampi::target_buffer<T> make_target_buffer(::yampi::displacement const displacement, int count, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count)))
  { return ::yampi::target_buffer<T>(displacement, count); }

  template <typename T>
  inline ::yampi::target_buffer<T> make_target_buffer(::yampi::displacement const displacement, int count, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::target_buffer<T>(displacement, count, datatype)))
  { return ::yampi::target_buffer<T>(displacement, count, datatype); }
}


# undef YAMPI_is_nothrow_swappable

#endif
