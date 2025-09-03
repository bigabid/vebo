import { Server, Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { useDataSources } from '@/hooks/useAthenaApi';

interface DataSourceSelectorProps {
  selectedDataSource: string | null;
  onChange: (dataSource: string) => void;
}

export function DataSourceSelector({ selectedDataSource, onChange }: DataSourceSelectorProps) {
  const { data, isLoading, error } = useDataSources();

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Server className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Data Source</h3>
          <p className="text-sm text-muted-foreground">Choose an Athena data source</p>
        </div>
      </div>

      {error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load data sources</p>
        </div>
      ) : (
        <SearchableSelect
          value={selectedDataSource}
          onChange={onChange}
          options={data?.dataSources || []}
          placeholder={isLoading ? 'Loading data sources...' : 'Select a data source'}
          loading={isLoading}
        />
      )}
    </Card>
  );
}


