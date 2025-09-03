import { Database, Loader2 } from 'lucide-react';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { useTables } from '@/hooks/useAthenaApi';
import { Card } from '@/components/ui/card';

interface TableSelectorProps {
  dataSource: string | null;
  catalog: string | null;
  database: string | null;
  selectedTable: string | null;
  onTableSelect: (table: string) => void;
}

export function TableSelector({ dataSource, catalog, database, selectedTable, onTableSelect }: TableSelectorProps) {
  const { data, isLoading, error } = useTables(dataSource, catalog, database);

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Database className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Table</h3>
          <p className="text-sm text-muted-foreground">Choose an Athena table to analyze</p>
        </div>
      </div>

      {(!dataSource || !catalog || !database) ? (
        <div className="p-4 bg-muted/50 rounded-lg border border-border">
          <p className="text-sm text-muted-foreground text-center">
            Select data source, catalog and database to list tables
          </p>
        </div>
      ) : error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load tables</p>
          <p className="text-xs text-destructive/80 mt-1">Please check your connection and try again</p>
        </div>
      ) : (
        <SearchableSelect
          value={selectedTable}
          onChange={onTableSelect}
          options={(data?.tables || []).map(t => ({ label: t.name, value: t.name }))}
          placeholder={isLoading ? 'Loading tables...' : 'Select a table to analyze'}
          loading={isLoading}
        />
      )}
    </Card>
  );
}