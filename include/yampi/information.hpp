#ifndef YAMPI_INFORMATION_HPP
# define YAMPI_INFORMATION_HPP

# include <string>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class information
  {
    MPI_Info mpi_info_;

   public:
    information() noexcept(std::is_nothrow_copy_constructible<MPI_Info>::value)
      : mpi_info_{MPI_INFO_NULL}
    { }

    information(information const&) = delete;
    information& operator=(information const&) = delete;

    information(information&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Info>::value
        and std::is_nothrow_copy_assignable<MPI_Info>::value)
      : mpi_info_{std::move(other.mpi_info_)}
    { other.mpi_info_ = MPI_INFO_NULL; }

    information& operator=(information&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Info>::value
        and std::is_nothrow_copy_assignable<MPI_Info>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_info_ != MPI_INFO_NULL)
          MPI_Info_free(std::addressof(mpi_info_));
        mpi_info_ = std::move(other.mpi_info_);
        other.mpi_info_ = MPI_INFO_NULL;
      } 
      return *this;
    }

    ~information() noexcept
    {
      if (mpi_info_ == MPI_INFO_NULL)
        return;

      MPI_Info_free(std::addressof(mpi_info_));
    }

    explicit information(MPI_Info const& mpi_info) noexcept(std::is_nothrow_copy_constructible<MPI_Group>::value)
      : mpi_info_{mpi_info}
    { }

    explicit information(::yampi::environment const& environment)
      : mpi_info_{create(environment)}
    { }

    information(information const& other, ::yampi::environment const& environment)
      : mpi_info_{duplicate(other, environment)}
    { }

   private:
    MPI_Info create(::yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Info_create(std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::create", environment);
    }

    MPI_Info duplicate(
      information const& other, ::yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Info_dup(other.mpi_info_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::duplicate", environment);
    }

   public:
    void reset(MPI_Info const& mpi_info, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = mpi_info;
    }

    void reset(information&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = std::move(other.mpi_info_);
      other.mpi_info_ = MPI_INFO_NULL;
    }

    void reset(::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = create(environment);
    }

    void reset(information const& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_info_ = duplicate(other, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_info_ == MPI_INFO_NULL)
        return;

      int const error_code = MPI_Info_free(std::addressof(mpi_info_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::free", environment);
    }

    bool is_null() const noexcept(noexcept(mpi_info_ == MPI_INFO_NULL))
    { return mpi_info_ == MPI_INFO_NULL; }

    void insert(
      std::string const& key, std::string const& value,
      ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      int const error_code = MPI_Info_set(mpi_info_, key.c_str(), value.c_str());
# else
      int const error_code
        = MPI_Info_set(
            mpi_info_, const_cast<char*>(key.c_str()), const_cast<char*>(value.c_str()));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::insert", environment);
    }

    void erase(std::string const& key, ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      int const error_code = MPI_Info_delete(mpi_info_, key.c_str());
# else
      int const error_code = MPI_Info_delete(mpi_info_, const_cast<char*>(key.c_str()));
# endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::erase", environment);
    }

    boost::optional<std::string> at(
      std::string const& key, ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 4
      auto buffer_length = 0;
      int flag;
      int error_code
        = MPI_Info_get_string(
            mpi_info_, key.c_str(),
            std::addressof(buffer_length), nullptr, std::addressof(flag));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::at", environment);

      if (flag == false)
        return boost::none;

      std::string result(buffer_length - 1, ' ');
      error_code
        = MPI_Info_get_string(
            mpi_info_, key.c_str(),
            std::addressof(buffer_length), const_cast<char*>(result.c_str()), std::addressof(flag));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::at", environment);

      if (flag == false)
        return boost::none;

      return boost::make_optional(result);
# else // MPI_VERSION >= 4
      int value_length;
      int flag;
#   if MPI_VERSION >= 3
      int error_code
        = MPI_Info_get_valuelen(
            mpi_info_, key.c_str(),
            std::addressof(value_length), std::addressof(flag));
#   else
      int error_code
        = MPI_Info_get_valuelen(
            mpi_info_, const_cast<char*>(key.c_str()),
            std::addressof(value_length), std::addressof(flag));
#   endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::at", environment);

      if (flag == false)
        return boost::none;

      std::string result(value_length, ' ');
#   if MPI_VERSION >= 3
      error_code
        = MPI_Info_get(
            mpi_info_, key.c_str(),
            value_length, const_cast<char*>(result.c_str()), std::addressof(flag));
#   else
      error_code
        = MPI_Info_get(
            mpi_info_, const_cast<char*>(key.c_str()),
            value_length, const_cast<char*>(result.c_str()), std::addressof(flag));
#   endif
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::information::at", environment);

      if (flag == false)
        return boost::none;

      return boost::make_optional(result);
# endif // MPI_VERSION >= 4
    }

    int num_keys(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Info_get_nkeys(mpi_info_, std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::information::num_keys", environment);
    }

    std::string key(int const n, ::yampi::environment const& environment) const
    {
      char key[MPI_MAX_INFO_KEY];
      int const error_code = MPI_Info_get_nthkey(mpi_info_, n, key);
      return error_code == MPI_SUCCESS
        ? std::string(key)
        : throw ::yampi::error(error_code, "yampi::information::key", environment);
    }

    MPI_Info const& mpi_info() const noexcept { return mpi_info_; }

    void swap(information& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Info>::value)
    {
      using std::swap;
      swap(mpi_info_, other.mpi_info_);
    }
  };

  inline void swap(::yampi::information& lhs, ::yampi::information& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

