import { useState } from 'react';
import { Play, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LogDisplay } from '@/components/LogDisplay';
import { useStartInsights, useJobStatus } from '@/hooks/useAthenaApi';
import { useToast } from '@/hooks/use-toast';

export interface ExecuteButtonComponentProps {
  dataSource: string | null;
  catalog: string | null;
  database: string | null;
  selectedTable: string | null;
  selectedPartitions: Record<string, string[]>;
  onInsightsReady: (insights: any) => void;
}

export function ExecuteButton({ 
  dataSource,
  catalog,
  database,
  selectedTable, 
  selectedPartitions, 
  onInsightsReady 
}: ExecuteButtonComponentProps) {
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const { toast } = useToast();
  
  const startInsightsMutation = useStartInsights();
  const jobStatusQuery = useJobStatus(currentJobId, !!currentJobId);

  const handleExecute = async () => {
    if (!dataSource || !catalog || !database || !selectedTable) return;
    
    try {
      const result = await startInsightsMutation.mutateAsync({
        dataSource,
        catalog,
        database,
        table: selectedTable,
        partitions: selectedPartitions,
      });
      
      setCurrentJobId(result.jobId);
      
      toast({
        title: "Analysis Started",
        description: `Running insights for ${selectedTable}`,
      });
    } catch (error) {
      toast({
        title: "Failed to start analysis",
        description: "Please try again",
        variant: "destructive",
      });
    }
  };

  const isRunning = jobStatusQuery.data?.status === 'running';
  const isComplete = jobStatusQuery.data?.status === 'complete';
  const isError = jobStatusQuery.data?.status === 'error';
  const progress = jobStatusQuery.data?.progress || 0;

  // Handle completion
  if (isComplete && jobStatusQuery.data?.insights) {
    if (currentJobId) {
      onInsightsReady(jobStatusQuery.data.insights);
      setCurrentJobId(null);
    }
  }

  // Handle error
  if (isError && currentJobId) {
    toast({
      title: "Analysis Failed",
      description: jobStatusQuery.data?.message || "Something went wrong",
      variant: "destructive",
    });
    setCurrentJobId(null);
  }

  const canExecute = !!dataSource && !!catalog && !!database && !!selectedTable && !isRunning && !startInsightsMutation.isPending;
  const selectedCount = Object.values(selectedPartitions).flat().length;

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-foreground">Execute Analysis</h3>
            <p className="text-sm text-muted-foreground">
              {selectedTable ? `Analyze ${selectedTable}` : 'Select a table to analyze'}
              {selectedCount > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedCount} partitions selected
                </Badge>
              )}
            </p>
          </div>
          
          {(isRunning || startInsightsMutation.isPending) && (
            <div className="flex items-center gap-2">
              {isComplete ? (
                <CheckCircle className="w-5 h-5 text-success" />
              ) : isError ? (
                <AlertCircle className="w-5 h-5 text-destructive" />
              ) : (
                <Loader2 className="w-5 h-5 animate-spin text-primary" />
              )}
            </div>
          )}
        </div>

        {(isRunning || isComplete) && jobStatusQuery.data?.logs && (
          <LogDisplay 
            logs={jobStatusQuery.data.logs}
            isRunning={isRunning}
            maxHeight="300px"
            className="mt-4"
          />
        )}

        <Button
          onClick={handleExecute}
          disabled={!canExecute}
          size="lg"
          className="w-full h-12 bg-gradient-primary hover:opacity-90 border-0 shadow-medium hover:shadow-glow transition-all"
        >
          {(isRunning || startInsightsMutation.isPending) ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              {startInsightsMutation.isPending ? 'Starting...' : 'Running Analysis...'}
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Execute Insights
            </>
          )}
        </Button>

        {selectedTable && (
          <div className="text-xs text-muted-foreground bg-muted/30 p-3 rounded-lg">
            <strong>What will be analyzed:</strong>
            <ul className="mt-1 space-y-1">
              <li>• Row count and partition statistics</li>
              <li>• Column profiling and data types</li>
              <li>• Null ratios and value distributions</li>
              <li>• Statistical summaries for numeric columns</li>
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}