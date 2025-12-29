import { ChatContext } from '@/app/chat-context';
import { LinkOutlined } from '@ant-design/icons';
import { useContext } from 'react';
import ExcelUpload from './excel-upload';

interface Props {
  onComplete?: () => void;
}

function ChatExcel({ onComplete }: Props) {
  const { currentDialogue, scene, chatId } = useContext(ChatContext);

  if (scene !== 'chat_excel') return null;

  // 只有当 currentDialogue 存在且 conv_uid 匹配当前 chatId 时才显示已上传文件
  // 这样可以确保新会话时不会显示上一个会话的文件信息
  const hasUploadedFile = currentDialogue?.conv_uid === chatId && currentDialogue?.select_param;

  return (
    <div className='max-w-md h-full relative'>
      {hasUploadedFile ? (
        <div className='flex h-8 overflow-hidden rounded'>
          <div className='flex items-center justify-center px-2 bg-gray-600 text-lg'>
            <LinkOutlined className='text-white' />
          </div>
          <div className='flex items-center justify-center px-3 bg-gray-100 text-xs rounded-tr rounded-br dark:text-gray-800 truncate'>
            {currentDialogue.select_param}
          </div>
        </div>
      ) : (
        <ExcelUpload convUid={chatId} chatMode={scene} onComplete={onComplete} />
      )}
    </div>
  );
}

export default ChatExcel;
