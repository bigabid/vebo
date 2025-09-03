import { useState, useRef, useEffect } from 'react';
import { ChevronDown, ChevronRight, Clock, Info, AlertTriangle, AlertCircle, Terminal } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import type { LogEntry } from '@/lib/api';

interface LogDisplayProps {
  logs: LogEntry[];
  isRunning?: boolean;
  className?: string;
  maxHeight?: string;
}

const LogLevelIcon = ({ level }: { level: string }) => {
  switch (level) {
    case 'info':
      return <Info className="w-4 h-4 text-blue-500" />;
    case 'warning':
      return <AlertTriangle className="w-4 h-4 text-amber-500" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-red-500" />;
    default:
      return <Terminal className="w-4 h-4 text-gray-500" />;
  }
};

const LogLevelBadge = ({ level }: { level: string }) => {
  const variants = {
    info: "default" as const,
    warning: "secondary" as const,
    error: "destructive" as const,
  };
  
  return (
    <Badge variant={variants[level as keyof typeof variants] || "outline"} className="text-xs">
      {level.toUpperCase()}
    </Badge>
  );
};

const formatTimestamp = (timestamp: string) => {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  } catch {
    return timestamp;
  }
};

const StageProgress = ({ logs }: { logs: LogEntry[] }) => {
  const stages = [
    'initialization',
    'athena_polling', 
    'data_download',
    'data_preparation',
    'sampling_decision',
    'column_analysis',
    'column_checks',
    'cross_column_checks',
    'table_checks', 
    'results_compilation',
    'post_processing'
  ];
  
  const completedStages = new Set(logs.map(log => log.stage));
  const currentStage = logs.length > 0 ? logs[logs.length - 1].stage : null;
  
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-2">
        <Terminal className="w-4 h-4" />
        <span className="text-sm font-medium">Processing Stages</span>
      </div>
      <div className="flex flex-wrap gap-1">
        {stages.map((stage, index) => {
          const isCompleted = completedStages.has(stage);
          const isCurrent = stage === currentStage;
          const stageLabel = stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          
          return (
            <Badge
              key={stage}
              variant={
                isCompleted && !isCurrent ? "default" : 
                isCurrent ? "secondary" : 
                "outline"
              }
              className={`text-xs ${isCurrent ? 'animate-pulse' : ''}`}
            >
              {stageLabel}
            </Badge>
          );
        })}
      </div>
    </div>
  );
};

interface LogEntryDisplayProps {
  log: LogEntry;
  isLatest?: boolean;
}

const LogEntryDisplay = ({ log, isLatest }: LogEntryDisplayProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasDetails = log.details && Object.keys(log.details).length > 0;
  
  return (
    <div 
      className={`flex gap-3 p-3 rounded-lg border transition-colors ${
        isLatest ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200'
      }`}
    >
      <div className="flex-shrink-0 mt-0.5">
        <LogLevelIcon level={log.level} />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <LogLevelBadge level={log.level} />
          <Badge variant="outline" className="text-xs">
            {log.stage.replace(/_/g, ' ')}
          </Badge>
          <div className="flex items-center gap-1 text-xs text-gray-500 ml-auto">
            <Clock className="w-3 h-3" />
            {formatTimestamp(log.timestamp)}
          </div>
        </div>
        
        <p className="text-sm text-gray-900 mb-2">{log.message}</p>
        
        {hasDetails && (
          <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
            <CollapsibleTrigger className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-800">
              {isExpanded ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
              View details
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              <pre className="text-xs bg-gray-100 p-2 rounded border overflow-x-auto">
                {JSON.stringify(log.details, null, 2)}
              </pre>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
};

export function LogDisplay({ logs, isRunning = false, className = "", maxHeight = "400px" }: LogDisplayProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  // Detect manual scrolling to disable auto-scroll
  const handleScroll = () => {
    if (!scrollRef.current) return;
    
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop === clientHeight;
    setAutoScroll(isAtBottom);
  };

  if (logs.length === 0) {
    return (
      <Card className={`p-6 ${className}`}>
        <div className="flex items-center justify-center text-gray-500">
          <Terminal className="w-5 h-5 mr-2" />
          <span>Waiting for processing to begin...</span>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`${className}`}>
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-5 h-5" />
            <h3 className="font-semibold">Processing Logs</h3>
            <Badge variant="outline" className="text-xs">
              {logs.length} entries
            </Badge>
          </div>
          {isRunning && (
            <Badge variant="secondary" className="animate-pulse">
              Running
            </Badge>
          )}
        </div>
      </div>
      
      <div className="p-4">
        <StageProgress logs={logs} />
        
        <ScrollArea 
          className="w-full border rounded-lg"
          style={{ height: maxHeight }}
        >
          <div 
            ref={scrollRef}
            className="p-4 space-y-3"
            onScroll={handleScroll}
          >
            {logs.map((log, index) => (
              <LogEntryDisplay
                key={`${log.timestamp}-${index}`}
                log={log}
                isLatest={index === logs.length - 1 && isRunning}
              />
            ))}
            
            {!autoScroll && (
              <button
                onClick={() => {
                  setAutoScroll(true);
                  if (scrollRef.current) {
                    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                  }
                }}
                className="fixed bottom-4 right-4 bg-blue-600 text-white px-3 py-1 rounded-full text-sm hover:bg-blue-700 transition-colors shadow-lg"
              >
                Scroll to bottom
              </button>
            )}
          </div>
        </ScrollArea>
      </div>
    </Card>
  );
}
