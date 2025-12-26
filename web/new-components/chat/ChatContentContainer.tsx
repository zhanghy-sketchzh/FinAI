import ChatHeader from '@/new-components/chat/header/ChatHeader';
import { ChatContentContext } from '@/pages/chat';
import { VerticalAlignBottomOutlined, VerticalAlignTopOutlined } from '@ant-design/icons';
import dynamic from 'next/dynamic';
import React, {
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';

const ChatCompletion = dynamic(() => import('@/new-components/chat/content/ChatCompletion'), { ssr: false });

// eslint-disable-next-line no-empty-pattern
const ChatContentContainer = ({ className }: { className?: string }, ref: React.ForwardedRef<any>) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isScrollToTop, setIsScrollToTop] = useState<boolean>(false);
  const [showScrollButtons, setShowScrollButtons] = useState<boolean>(false);
  const [isAtTop, setIsAtTop] = useState<boolean>(true);
  const [isAtBottom, setIsAtBottom] = useState<boolean>(false);
  const { history } = useContext(ChatContentContext);
  const allowAutoScroll = useRef<boolean>(true);
  const animationFrameRef = useRef<number | null>(null);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastScrollHeightRef = useRef<number>(0);

  useImperativeHandle(ref, () => {
    return scrollRef.current;
  });

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;

    const container = scrollRef.current;
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    const buffer = 20;

    // Check Scroll direction
    const lastScrollTop = Number(container?.dataset?.lastScrollTop) || 0;
    const direction = scrollTop > lastScrollTop ? 'down' : 'up';
    container.dataset.lastScrollTop = String(scrollTop);
    
    // Only disable auto scroll if user manually scrolls up significantly
    // If user is near bottom or scrolling down, allow auto scroll
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - buffer * 2;
    allowAutoScroll.current = direction === 'down' || isNearBottom;

    // Check if we're at the top
    setIsAtTop(scrollTop <= buffer);

    // Check if we're at the bottom
    setIsAtBottom(scrollTop + clientHeight >= scrollHeight - buffer);

    // Header visibility
    if (scrollTop >= 42 + 32) {
      setIsScrollToTop(true);
    } else {
      setIsScrollToTop(false);
    }

    // Show scroll buttons when content is scrollable
    const isScrollable = scrollHeight > clientHeight;
    setShowScrollButtons(isScrollable);
  }, []);

  useEffect(() => {
    const currentScrollRef = scrollRef.current;
    if (currentScrollRef) {
      currentScrollRef.addEventListener('scroll', handleScroll);

      // Check initially if content is scrollable
      const isScrollable = currentScrollRef.scrollHeight > currentScrollRef.clientHeight;
      setShowScrollButtons(isScrollable);
    }

    return () => {
      if (currentScrollRef) {
        currentScrollRef.removeEventListener('scroll', handleScroll);
      }
    };
  }, [handleScroll]);
  const scrollToBottomSmooth = useCallback((forceScroll = false) => {
    if (!scrollRef.current) return;

    // For force scroll (new messages), bypass allowAutoScroll check
    if (!forceScroll && !allowAutoScroll.current) return;

    const container = scrollRef.current;
    const { scrollTop, scrollHeight, clientHeight } = container;

    // Only auto-scroll when user is near bottom, unless force scroll is requested
    const buffer = Math.max(50, clientHeight * 0.1);
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - buffer;

    if (!isNearBottom && !forceScroll) {
      return;
    }

    // Clear previous timeouts and animation frames
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
      scrollTimeoutRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    // Use a combination of requestAnimationFrame and setTimeout to ensure DOM is updated
    const performScroll = () => {
      if (scrollRef.current) {
        const targetScrollHeight = scrollRef.current.scrollHeight;
        scrollRef.current.scrollTo({
          top: targetScrollHeight,
          behavior: forceScroll ? 'smooth' : 'auto',
        });
        // Also try direct assignment as fallback
        if (scrollRef.current.scrollTop < targetScrollHeight - 10) {
          scrollRef.current.scrollTop = targetScrollHeight;
        }
        // Update last scroll height
        lastScrollHeightRef.current = targetScrollHeight;
      }
    };

    // Use requestAnimationFrame for immediate scroll
    animationFrameRef.current = requestAnimationFrame(() => {
      performScroll();
      // Also set multiple timeouts to ensure scroll happens after DOM updates
      scrollTimeoutRef.current = setTimeout(() => {
        performScroll();
        // One more attempt after a longer delay
        setTimeout(() => {
          if (scrollRef.current) {
            const currentScrollHeight = scrollRef.current.scrollHeight;
            if (currentScrollHeight !== lastScrollHeightRef.current) {
              performScroll();
            }
          }
        }, 100);
        scrollTimeoutRef.current = null;
      }, 50);
      animationFrameRef.current = null;
    });
  }, []);

  // Optimize last message tracking to reduce unnecessary re-renders
  const lastMessage = useMemo(() => {
    const last = history[history.length - 1];
    return last ? { context: last.context, thinking: last.thinking, role: last.role } : null;
  }, [history]);
  const prevHistoryLengthRef = useRef(history.length);
  const prevLastMessageRef = useRef<string>('');

  useEffect(() => {
    const currentHistoryLength = history.length;
    const isNewMessage = currentHistoryLength > prevHistoryLengthRef.current;
    const lastMessageKey = lastMessage ? `${lastMessage.role}-${lastMessage.context}-${lastMessage.thinking}` : '';
    const isMessageChanged = lastMessageKey !== prevLastMessageRef.current;

    if (isNewMessage) {
      // Force scroll to bottom when new message is added
      // Reset allowAutoScroll to true for new messages
      allowAutoScroll.current = true;
      prevHistoryLengthRef.current = currentHistoryLength;
      prevLastMessageRef.current = lastMessageKey;
      
      // Use multiple scroll attempts to ensure it works
      setTimeout(() => {
        scrollToBottomSmooth(true);
      }, 0);
      setTimeout(() => {
        scrollToBottomSmooth(true);
      }, 100);
      setTimeout(() => {
        scrollToBottomSmooth(true);
      }, 300);
    } else if (isMessageChanged) {
      // Message content changed (streaming update)
      prevLastMessageRef.current = lastMessageKey;
      
      // Scroll during streaming updates
      setTimeout(() => {
        scrollToBottomSmooth(false);
      }, 0);
      setTimeout(() => {
        scrollToBottomSmooth(false);
      }, 100);
    }
  }, [history.length, lastMessage?.context, lastMessage?.thinking, lastMessage?.role, scrollToBottomSmooth]);

  // Use ResizeObserver and MutationObserver to detect when content changes
  useEffect(() => {
    if (!scrollRef.current) return;

    const container = scrollRef.current;
    let scrollTimer: NodeJS.Timeout | null = null;

    const checkAndScroll = () => {
      if (!container) return;
      
      const currentScrollHeight = container.scrollHeight;
      const hasNewContent = currentScrollHeight !== lastScrollHeightRef.current;
      
      if (hasNewContent) {
        lastScrollHeightRef.current = currentScrollHeight;
        
        // Always scroll if auto-scroll is enabled or if user is near bottom
        if (allowAutoScroll.current) {
          // Use multiple attempts to ensure scroll happens after DOM updates
          scrollTimer = setTimeout(() => {
            if (container) {
              container.scrollTo({
                top: container.scrollHeight,
                behavior: 'smooth',
              });
            }
          }, 50);
          
          // Also try immediate scroll
          requestAnimationFrame(() => {
            if (container) {
              container.scrollTo({
                top: container.scrollHeight,
                behavior: 'auto',
              });
            }
          });
        }
      }
    };

    // ResizeObserver for size changes
    const resizeObserver = new ResizeObserver(() => {
      checkAndScroll();
    });

    // MutationObserver for DOM changes (new messages added)
    const mutationObserver = new MutationObserver(() => {
      checkAndScroll();
    });

    resizeObserver.observe(container);
    mutationObserver.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    // Also check periodically during streaming
    const intervalId = setInterval(() => {
      checkAndScroll();
    }, 200);

    return () => {
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      clearInterval(intervalId);
      if (scrollTimer) {
        clearTimeout(scrollTimer);
      }
    };
  }, []);

  // Cleanup animation frame and timeout on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);
  const scrollToTop = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, []);

  return (
    <div className={`flex flex-1 overflow-hidden relative ${className || ''}`}>
      <div ref={scrollRef} className='h-full w-full mx-auto overflow-y-auto'>
        <ChatHeader isScrollToTop={isScrollToTop} />
        <ChatCompletion />
      </div>

      {showScrollButtons && (
        <div className='absolute right-4 md:right-6 bottom-[120px] md:bottom-[100px] flex flex-col gap-2 z-[999]'>
          {!isAtTop && (
            <button
              onClick={scrollToTop}
              className='w-9 h-9 md:w-10 md:h-10 bg-white dark:bg-[rgba(255,255,255,0.2)] border border-gray-200 dark:border-[rgba(255,255,255,0.2)] rounded-full flex items-center justify-center shadow-md hover:shadow-lg transition-all duration-200'
              aria-label='Scroll to top'
            >
              <VerticalAlignTopOutlined className='text-[#525964] dark:text-[rgba(255,255,255,0.85)] text-sm md:text-base' />
            </button>
          )}
          {!isAtBottom && (
            <button
              onClick={scrollToBottom}
              className='w-9 h-9 md:w-10 md:h-10 bg-white dark:bg-[rgba(255,255,255,0.2)] border border-gray-200 dark:border-[rgba(255,255,255,0.2)] rounded-full flex items-center justify-center shadow-md hover:shadow-lg transition-all duration-200'
              aria-label='Scroll to bottom'
            >
              <VerticalAlignBottomOutlined className='text-[#525964] dark:text-[rgba(255,255,255,0.85)] text-sm md:text-base' />
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default forwardRef(ChatContentContainer);
