import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Cloud, Database, TrendingUp } from 'lucide-react';
import { Toaster } from '@/components/ui/toaster';
import { TableSelector } from '@/components/TableSelector';
import { PartitionSelector } from '@/components/PartitionSelector';
import { ExecuteButton } from '@/components/ExecuteButton';
import { InsightsPanel } from '@/components/InsightsPanel';
import type { InsightsData } from '@/lib/api';

const queryClient = new QueryClient();

function AthenaExplorer() {
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [selectedPartitions, setSelectedPartitions] = useState<Record<string, string[]>>({});
  const [currentInsights, setCurrentInsights] = useState<InsightsData | null>(null);

  const handleTableSelect = (table: string) => {
    setSelectedTable(table);
    setSelectedPartitions({});
    setCurrentInsights(null);
  };

  const handleInsightsReady = (insights: InsightsData) => {
    setCurrentInsights(insights);
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Cloud className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">Athena Insight Explorer</h1>
              <p className="text-sm text-muted-foreground">Analyze your AWS Athena tables with advanced insights</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="xl:col-span-1 space-y-6">
            <TableSelector
              selectedTable={selectedTable}
              onTableSelect={handleTableSelect}
            />
            
            <PartitionSelector
              selectedTable={selectedTable}
              selectedPartitions={selectedPartitions}
              onPartitionChange={setSelectedPartitions}
            />
            
            <ExecuteButton
              selectedTable={selectedTable}
              selectedPartitions={selectedPartitions}
              onInsightsReady={handleInsightsReady}
            />
          </div>

          {/* Right Panel - Results */}
          <div className="xl:col-span-2">
            <InsightsPanel insights={currentInsights} />
          </div>
        </div>
      </main>
    </div>
  );
}

const Index = () => (
  <QueryClientProvider client={queryClient}>
    <AthenaExplorer />
    <Toaster />
  </QueryClientProvider>
);

export default Index;
