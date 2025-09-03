import { Database as DbIcon, Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { SearchableSelect } from '@/components/ui/searchable-select';
import { useDatabases } from '@/hooks/useAthenaApi';

interface DatabaseSelectorProps {
  dataSource: string | null;
  catalog: string | null;
  selectedDatabase: string | null;
  onChange: (database: string) => void;
}

export function DatabaseSelector({ dataSource, catalog, selectedDatabase, onChange }: DatabaseSelectorProps) {
  const { data, isLoading, error } = useDatabases(dataSource, catalog);

  if (!dataSource || !catalog) return null;

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-secondary/10 rounded-lg">
          <DbIcon className="w-5 h-5 text-secondary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Database</h3>
          <p className="text-sm text-muted-foreground">Choose a database in the catalog</p>
        </div>
      </div>

      {error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load databases</p>
        </div>
      ) : (
        <SearchableSelect
          value={selectedDatabase}
          onChange={onChange}
          options={data?.databases || []}
          placeholder={isLoading ? 'Loading databases...' : 'Select a database'}
          loading={isLoading}
        />
      )}
    </Card>
  );
}


