import { UserInfoResponse } from '@/types/userinfo';
import { STORAGE_USERINFO_KEY } from '@/utils/constants/index';
import { Avatar } from 'antd';
import cls from 'classnames';
import { useEffect, useState } from 'react';

function UserBar({ onlyAvatar = false, align = 'center' }: { onlyAvatar?: boolean; align?: 'left' | 'center' }) {
  const [userInfo, setUserInfo] = useState<UserInfoResponse>();
  useEffect(() => {
    try {
      const user = JSON.parse(localStorage.getItem(STORAGE_USERINFO_KEY) ?? '');
      setUserInfo(user);
    } catch {
      return undefined;
    }
  }, []);

  // TODO: delete unused function
  // const logout = () => {
  //   localStorage.removeItem(STORAGE_USERINFO_KEY);
  //   window.location.href = `${process.env.LOGOUT_URL}&goto=${encodeURIComponent(window.location.href)}`;
  // };

  return (
    <div className={cls('flex items-center', { 'flex-1 justify-center': align === 'center' })}>
      <div
        className={cls('flex items-center group w-full', {
          'justify-center': onlyAvatar && align === 'center',
          'justify-between': !onlyAvatar,
        })}
      >
        <span className='flex gap-2 items-center'>
          <Avatar src={userInfo?.avatar_url} className='bg-gradient-to-tr from-[#31afff] to-[#1677ff] cursor-pointer'>
            user
          </Avatar>
          <span
            className={cls('text-sm', {
              hidden: onlyAvatar,
            })}
          >
            {userInfo?.nick_name}
          </span>
        </span>
        {/* <LogoutOutlined
          onClick={logout}
          className={cls('cursor-pointer opacity-0 transition-all hover:opacity-100 group-hover:opacity-70', {
            hidden: onlyAvatar,
          })}
        /> */}
      </div>
    </div>
  );
}

export default UserBar;
