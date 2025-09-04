import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Cloud } from 'lucide-react';
import { Toaster } from '@/components/ui/toaster';
import { TableSelector } from '@/components/TableSelector';
import { PartitionSelector } from '@/components/PartitionSelector';
import { ExecuteButton } from '@/components/ExecuteButton';
import { InsightsPanel } from '@/components/InsightsPanel';
import { QueryEditor } from '@/components/QueryEditor';
import type { InsightsData } from '@/lib/api';
import { DataSourceSelector } from '@/components/DataSourceSelector';
import { CatalogSelector } from '@/components/CatalogSelector';
import { DatabaseSelector } from '@/components/DatabaseSelector';

const queryClient = new QueryClient();

function AthenaExplorer() {
  const [selectedDataSource, setSelectedDataSource] = useState<string | null>('AwsDataCatalog');
  const [selectedCatalog, setSelectedCatalog] = useState<string | null>('AwsDataCatalog');
  const [selectedDatabase, setSelectedDatabase] = useState<string | null>('dmp_data');
  const [selectedTable, setSelectedTable] = useState<string | null>('parquet_bids');
  const [selectedPartitions, setSelectedPartitions] = useState<Record<string, string[]>>({});
  const [currentInsights, setCurrentInsights] = useState<InsightsData | null>(null);
  const [currentQuery, setCurrentQuery] = useState<string>('');

  const handleDataSourceSelect = (ds: string) => {
    setSelectedDataSource(ds);
    setSelectedCatalog(null);
    setSelectedDatabase(null);
    setSelectedTable(null);
    setSelectedPartitions({});
    setCurrentInsights(null);
  };

  const handleCatalogSelect = (catalog: string) => {
    setSelectedCatalog(catalog);
    setSelectedDatabase(null);
    setSelectedTable(null);
    setSelectedPartitions({});
    setCurrentInsights(null);
  };

  const handleDatabaseSelect = (db: string) => {
    setSelectedDatabase(db);
    setSelectedTable(null);
    setSelectedPartitions({});
    setCurrentInsights(null);
  };

  const handleTableSelect = (table: string) => {
    setSelectedTable(table);
    setSelectedPartitions({});
    setCurrentInsights(null);
  };

  const handleInsightsReady = (insights: InsightsData) => {
    setCurrentInsights(insights);
  };

  const handleQueryChange = (query: string) => {
    setCurrentQuery(query);
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
            <DataSourceSelector
              selectedDataSource={selectedDataSource}
              onChange={handleDataSourceSelect}
            />

            <CatalogSelector
              dataSource={selectedDataSource}
              selectedCatalog={selectedCatalog}
              onChange={handleCatalogSelect}
            />

            <DatabaseSelector
              dataSource={selectedDataSource}
              catalog={selectedCatalog}
              selectedDatabase={selectedDatabase}
              onChange={handleDatabaseSelect}
            />

            <TableSelector
              dataSource={selectedDataSource}
              catalog={selectedCatalog}
              database={selectedDatabase}
              selectedTable={selectedTable}
              onTableSelect={handleTableSelect}
            />
            
            {false && (
              <PartitionSelector
                catalog={selectedCatalog}
                database={selectedDatabase}
                selectedTable={selectedTable}
                selectedPartitions={selectedPartitions}
                onPartitionChange={setSelectedPartitions}
              />
            )}
            
            <QueryEditor
              dataSource={selectedDataSource}
              catalog={selectedCatalog}
              database={selectedDatabase}
              table={selectedTable}
              partitions={selectedPartitions}
              onQueryChange={handleQueryChange}
            />
            
            <ExecuteButton
              dataSource={selectedDataSource}
              catalog={selectedCatalog}
              database={selectedDatabase}
              selectedTable={selectedTable}
              selectedPartitions={selectedPartitions}
              currentQuery={currentQuery}
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
